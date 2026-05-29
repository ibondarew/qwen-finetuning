import gc
import os
import pathlib

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Оптимизация аллокатора памяти CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = str(pathlib.Path(__file__).parent.absolute())
print(f"Working dir: {BASE_DIR}")

# LoRA вешается строго на блоки внимания, чтобы не ломать 3D-тензоры экспертов MoE
Qwen3_CODER_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Дефолтная модель (подставится, если не задана переменная окружения MODEL_ID)
DEFAULT_MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507"


def train():
    # Инициализируем распределенный контекст (необходимо для работы барьеров)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"Local rank: {local_rank} initialized.")

    # Динамически получаем имя модели из окружения
    MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
    if local_rank == 0:
        print(f"\n=== TARGET MODEL FOR TRAINING: {MODEL_ID} ===\n")

    # Токенизатор загружаем на всех процессах (он легкий, кэш блокируется безопасно)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.empty_cache()
    gc.collect()

    # Общая конфигурация LoRA для всех процессов
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=Qwen3_CODER_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- ЗАЩИТА RAM: БАРЬЕР СИНХРОНИЗАЦИИ ---
    # Сначала модель создает только rank 0, чтобы скачать конфиги и прогреть кэш хоста
    if local_rank == 0:
        print("Master process (rank 0) is preparing the model structure...")
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        with torch.device("cpu"):
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
        model = get_peft_model(model, peft_config)
        model.gradient_checkpointing_enable()
        print("Master process successfully prepared the model framework.")

    # Все процессы (1-7) останавливаются тут и ждут, пока rank 0 завершит работу с диском
    dist.barrier()

    # Теперь, когда rank 0 всё подготовил, остальные процессы создают модель из локального кэша.
    # Это исключает race condition файлов и пиковые перегрузки RAM.
    if local_rank != 0:
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        with torch.device("cpu"):
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
        model = get_peft_model(model, peft_config)
        model.gradient_checkpointing_enable()

    # Финальная точка синхронизации: все процессы гарантированно имеют одинаковый пустой каркас модели
    dist.barrier()
    # --- КОНЕЦ БЛОКА ЗАЩИТЫ ---

    if local_rank == 0:
        model.print_trainable_parameters()

    torch.cuda.empty_cache()
    gc.collect()

    print(f"Process {local_rank} is loading dataset...")
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    dataset = dataset.filter(lambda x: x["instruction"] and x["output"], batched=False)

    def tokenize_function(examples):
        texts = []
        for inst, inp, out in zip(
            examples["instruction"], examples.get("input", [""]), examples["output"]
        ):
            if inp:
                text = f"Instruction: {inst}\nContext: {inp}\nResponse: {out}"
            else:
                text = f"Instruction: {inst}\nResponse: {out}"
            texts.append(text)

        result = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        result["labels"] = result["input_ids"].copy()
        return result

    # Токенизацию запускаем параллельно
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=256,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
        num_proc=1,
    )

    # Динамически выставляем имя папки чекпоинтов на основе названия модели
    model_folder_name = MODEL_ID.split("/")[-1].lower()
    training_args = TrainingArguments(
        output_dir=f"{BASE_DIR}/{model_folder_name}-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        bf16=True,
        logging_steps=1,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        max_steps=100,
        deepspeed=f"{BASE_DIR}/ds_config.json",
        report_to="none",
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        ),
    )

    if local_rank == 0:
        print(
            "Starting training process (DeepSpeed Stage 3 is taking over to stream weights)..."
        )

    trainer.train()


if __name__ == "__main__":
    train()

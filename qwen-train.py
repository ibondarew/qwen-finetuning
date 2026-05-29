import datetime
import gc
import os
import pathlib

import deepspeed
import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
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

# LoRA вешается строго на блоки внимания, чтобы не ломать и не трогать MoE экспертов
Qwen3_CODER_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Модель по умолчанию, если не передана переменная окружения
DEFAULT_MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507"


def train():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Явно привязываем GPU к процессу и выставляем таймаут в 30 минут для стабильного стриминга весов
    if not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=datetime.timedelta(seconds=1800),
        )
    print(f"Local rank: {local_rank} initialized.")

    # Динамически получаем имя модели из окружения
    MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
    if local_rank == 0:
        print(f"\n=== TARGET MODEL FOR TRAINING: {MODEL_ID} ===\n")

    # Токенизатор загружаем на всех процессах
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

    # ФИКС ОДИНОЧНОГО OOM И META TENSOR:
    # Возвращаем контекст ZeRO-3 Init, чтобы модель рождалась сразу секционированной по всем 8 GPU.
    # КРИТИЧЕСКИ ВАЖНО: Убрали параметр low_cpu_mem_usage=True. Именно он ломал совместимость с PEFT.
    print(f"Process {local_rank} is initializing model via DeepSpeed ZeRO-3 context...")
    with deepspeed.zero.Init():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # Теперь базовые веса корректно обернуты в ZeRO-3, и PEFT спокойно натянет LoRA слои
    print(f"Process {local_rank} is applying LoRA layers...")
    model = get_peft_model(model, peft_config)

    # Включаем чекпоинтинг градиентов для экономии памяти
    model.gradient_checkpointing_enable()

    print(f"Model and LoRA layers successfully prepared on rank {local_rank}")

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
        print("Starting training process (DeepSpeed Stage 3 is taking over)...")

    trainer.train()

    # Корректно закрываем распределенную группу при выходе
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()

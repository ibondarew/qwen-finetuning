import datetime
import gc
import json
import os
import pathlib

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

# Импортируем мост между Transformers и DeepSpeed
from transformers.integrations import HfDeepSpeedConfig

# Оптимизация аллокатора памяти CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = str(pathlib.Path(__file__).parent.absolute())
print(f"Working dir: {BASE_DIR}")

# LoRA вешается строго на блоки внимания
Qwen3_CODER_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507"


def train():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=datetime.timedelta(seconds=1800),
        )

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    print(f"Local rank: {local_rank} initialized. World size: {world_size}")

    MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
    if local_rank == 0:
        print(f"\n=== TARGET MODEL FOR TRAINING: {MODEL_ID} ===\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.empty_cache()
    gc.collect()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=Qwen3_CODER_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ДИНАМИЧЕСКАЯ СБОРКА КОНФИГА DEEPSPEED
    DS_CONFIG_PATH = os.path.join(BASE_DIR, "ds_config.json")
    with open(DS_CONFIG_PATH, "r") as f:
        ds_config = json.load(f)

    micro_batch = 1
    grad_accum = 16
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch
    ds_config["gradient_accumulation_steps"] = grad_accum
    ds_config["train_batch_size"] = micro_batch * grad_accum * world_size

    if local_rank == 0:
        print(
            f"Generated DS Config: train_batch_size={ds_config['train_batch_size']} for world_size={world_size}"
        )
        print(f"Process {local_rank} is initializing HfDeepSpeedConfig...")

    # ВАЖНО: Инициализируем HfDeepSpeedConfig ДО загрузки модели.
    # Обязательно сохраняем ссылку в переменную ds_helper, чтобы объект жил в памяти.
    # Это перехватит создание мета-тензоров внутри .from_pretrained() и предотвратит ошибку копирования.
    ds_helper = HfDeepSpeedConfig(ds_config)

    # Контекст "with deepspeed.zero.Init()" убираем — хелпер сделает всё сам автоматически и корректно
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Process {local_rank} is applying LoRA layers...")
    model = get_peft_model(model, peft_config)
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

    model_folder_name = MODEL_ID.split("/")[-1].lower()
    training_args = TrainingArguments(
        output_dir=f"{BASE_DIR}/{model_folder_name}-lora",
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=5e-6,
        bf16=True,
        logging_steps=1,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        max_steps=100,
        deepspeed=ds_config,
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

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()

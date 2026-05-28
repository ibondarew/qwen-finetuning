import gc
import os
import pathlib

import deepspeed
import torch
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

# Целевые модули для Qwen MoE архитектуры
Qwen3_CODER_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

MODEL_ID = "Qwen/Qwen3-Coder-480B-A35B-Instruct"


def train():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"Local rank: {local_rank}")

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # ФИКС МИСМАТЧА: У Qwen3 уже есть встроенный pad_token (<|endoftext|> или аналогичный).
    # Мы НЕ переназначаем его вручную на eos_token, чтобы Trainer не пытался вызвать
    # resize_token_embeddings, ломая Sharding Plan в DeepSpeed Stage 3.
    if tokenizer.pad_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.empty_cache()
    gc.collect()

    print("Loading 480B model with DeepSpeed ZeRO-3 Init...")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=Qwen3_CODER_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 1. Загружаем базовую модель на мета-девайс через ZeRO-3
    with deepspeed.zero.Init():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    # 2. РАБОЧИЙ ХАК: Временно отключаем флаг ZeRO-3 для корректной инициализации LoRA
    is_zero3_enabled = deepspeed.zero.Init.is_zero3_enabled
    deepspeed.zero.Init.is_zero3_enabled = lambda: False

    try:
        # Навешиваем LoRA слои поверх мета-структуры базовой модели
        model = get_peft_model(model, peft_config)
    finally:
        # Возвращаем флаг обратно, чтобы DeepSpeed взял управление на этапе обучения
        deepspeed.zero.Init.is_zero3_enabled = is_zero3_enabled

    print(f"Model and LoRA layers successfully initialized on rank {local_rank}")

    if local_rank == 0:
        model.print_trainable_parameters()

    torch.cuda.empty_cache()
    gc.collect()

    print("Loading dataset...")
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

        # Ограничиваем длину для экономии памяти на одном сервере
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

    training_args = TrainingArguments(
        output_dir=f"{BASE_DIR}/qwen3-coder-lora",
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

    print("Starting training process...")
    trainer.train()


if __name__ == "__main__":
    train()

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from preprocess import load_dailydialog, build_pairs_dataset

# ---------- 1. Данные ----------
print("Загрузка данных...")
ds = load_dailydialog()
train_pairs = build_pairs_dataset(ds, "train")
val_pairs = build_pairs_dataset(ds, "validation")

# Подвыборка, чтобы обучение шло минуты, а не часы
TRAIN_SIZE = 3000
VAL_SIZE = 200
train_pairs = train_pairs[:TRAIN_SIZE]
val_pairs = val_pairs[:VAL_SIZE]
print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

# ---------- 2. Токенизатор и формат ----------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LEN = 256

def format_and_tokenize(example):
    """
    Формат: <|user|>\n{user}</s>\n<|assistant|>\n{system}</s>
    Маскируем лосс на промпте: -100 для токенов user + assistant-заголовка,
    считаем лосс только на токенах ответа системы.
    """
    prompt = f"<|user|>\n{example['user']}</s>\n<|assistant|>\n"
    answer = f"{example['system']}</s>"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    # Обрезаем до MAX_LEN
    input_ids = input_ids[:MAX_LEN]
    labels = labels[:MAX_LEN]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }

train_ds = Dataset.from_list(train_pairs).map(format_and_tokenize, remove_columns=["user", "system"])
val_ds = Dataset.from_list(val_pairs).map(format_and_tokenize, remove_columns=["user", "system"])
print("Пример токенизации:", train_ds[0]["input_ids"][:20], "...")

# ---------- 3. Модель в 4-bit + LoRA ----------
print("Загрузка TinyLlama (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto", dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------- 4. Коллатор с паддингом ----------
def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ---------- 5. Обучение ----------
training_args = TrainingArguments(
    output_dir="./llama_lora_ckpt",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=50,
    logging_steps=20,
    save_strategy="no",
    eval_strategy="no",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=collate_fn,
)

print("Старт обучения...")
trainer.train()

# ---------- 6. Сохраняем LoRA-адаптер ----------
model.save_pretrained("./llama_lora_adapter")
tokenizer.save_pretrained("./llama_lora_adapter")
print("Адаптер сохранён в ./llama_lora_adapter")

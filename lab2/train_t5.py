import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from preprocess import load_dailydialog, build_pairs_dataset

# ---------- 1. Данные ----------
print("Загрузка данных...")
ds = load_dailydialog()
train_pairs = build_pairs_dataset(ds, "train")[:5000]
val_pairs = build_pairs_dataset(ds, "validation")[:200]
print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

# ---------- 2. Токенизация ----------
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
MAX_IN, MAX_OUT = 128, 64

def tokenize(example):
    model_inputs = tokenizer(
        example["user"], max_length=MAX_IN, truncation=True, padding=False,
    )
    labels = tokenizer(
        text_target=example["system"], max_length=MAX_OUT, truncation=True, padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_ds = Dataset.from_list(train_pairs).map(tokenize, remove_columns=["user", "system"])
val_ds = Dataset.from_list(val_pairs).map(tokenize, remove_columns=["user", "system"])

# ---------- 3. Модель ----------
print("Загрузка t5-small...")
model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda")

# ---------- 4. Коллатор ----------
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# ---------- 5. Обучение ----------
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_ckpt",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    warmup_steps=200,
    logging_steps=50,
    save_strategy="no",
    eval_strategy="no",
    fp16=True,
    optim="adamw_torch",
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=data_collator,
)

print("Старт обучения t5-small...")
trainer.train()

# ---------- 6. Сохраняем ----------
model.save_pretrained("./t5_finetuned")
tokenizer.save_pretrained("./t5_finetuned")
print("Модель сохранена в ./t5_finetuned")

import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from preprocess import load_dailydialog, build_pairs_dataset
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------- 1. Те же 100 пар из validation ----------
ds = load_dailydialog()
val_pairs = build_pairs_dataset(ds, "validation")[:100]
users = [p["user"] for p in val_pairs]
golds = [p["system"] for p in val_pairs]
print(f"Взято {len(users)} пар (те же, что в evaluate_before.py).")

# ---------- 2. Метрики ----------
def compute_metrics(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = nltk.word_tokenize(ref.lower())
        pred_tokens = nltk.word_tokenize(pred.lower())
        meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
    meteor = sum(meteor_scores) / len(meteor_scores)
    return bleu, meteor

# ---------- 3. Загрузка базовой модели + LoRA ----------
print("\n>>> Загрузка TinyLlama (4-bit) + LoRA-адаптер...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained("./llama_lora_adapter")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto", dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, "./llama_lora_adapter")
model.eval()
print("Модель с адаптером загружена.")

# ---------- 4. Генерация ----------
def llama_generate(user_text, max_new_tokens=64):
    prompt = f"<|user|>\n{user_text}</s>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("Генерация после обучения...")
preds = [llama_generate(u) for u in tqdm(users)]

# Показываем несколько примеров глазами
print("\n=== Примеры после дообучения ===")
for i in range(3):
    print(f"[{i}] USER:   {users[i]}")
    print(f"    GEN:    {preds[i]}")
    print(f"    GOLD:   {golds[i]}\n")

bleu, meteor = compute_metrics(preds, golds)
print(f"TinyLlama (после LoRA): BLEU={bleu:.2f}, METEOR={meteor:.4f}")

# ---------- 5. Сохраняем ----------
with open("metrics_llama_after.json", "w") as f:
    json.dump({"bleu": bleu, "meteor": meteor, "n_samples": len(users)}, f, indent=2)
print("\nСохранено в metrics_llama_after.json")

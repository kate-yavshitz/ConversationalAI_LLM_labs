import torch
import json
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
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

# ---------- 3. Загрузка дообученной t5-small ----------
print("\n>>> Загрузка t5-small из ./t5_finetuned...")
tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned")
model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned").to("cuda")
model.eval()
print("Модель загружена.")

# ---------- 4. Генерация ----------
def t5_generate(user_text, max_new_tokens=64):
    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

print("Генерация после обучения...")
preds = [t5_generate(u) for u in tqdm(users)]

# Показываем несколько примеров
print("\n=== Примеры после дообучения ===")
for i in range(3):
    print(f"[{i}] USER:   {users[i]}")
    print(f"    GEN:    {preds[i]}")
    print(f"    GOLD:   {golds[i]}\n")

bleu, meteor = compute_metrics(preds, golds)
print(f"t5-small (после дообучения): BLEU={bleu:.2f}, METEOR={meteor:.4f}")

# ---------- 5. Сохраняем ----------
with open("metrics_t5_after.json", "w") as f:
    json.dump({"bleu": bleu, "meteor": meteor, "n_samples": len(users)}, f, indent=2)
print("\nСохранено в metrics_t5_after.json")

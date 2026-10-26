import torch
import json
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    T5Tokenizer, T5ForConditionalGeneration,
    BitsAndBytesConfig,
)
from preprocess import load_dailydialog, build_pairs_dataset
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score

# Скачиваем ресурсы для METEOR (если ещё нет)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------- 1. Данные: 100 пар из validation ----------
print("Загрузка validation...")
ds = load_dailydialog()
val_pairs = build_pairs_dataset(ds, "validation")[:100]
users = [p["user"] for p in val_pairs]
golds = [p["system"] for p in val_pairs]
print(f"Взято {len(users)} пар.")

# ---------- 2. Функции подсчёта метрик ----------
def compute_metrics(predictions, references):
    """Возвращает BLEU (0-100) и METEOR (0-1)."""
    # BLEU через sacrebleu (возвращает 0-100)
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score

    # METEOR через nltk (усредняем по парам)
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        # nltk ожидает токенизированные списки
        ref_tokens = nltk.word_tokenize(ref.lower())
        pred_tokens = nltk.word_tokenize(pred.lower())
        meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
    meteor = sum(meteor_scores) / len(meteor_scores)

    return bleu, meteor

# ---------- 3. TinyLlama ----------
print("\n>>> Загрузка TinyLlama (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
llama_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
llama_tok = AutoTokenizer.from_pretrained(llama_name)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_name, quantization_config=bnb_config, device_map="auto", dtype=torch.float16,
)
llama_model.eval()

def llama_generate(user_text, max_new_tokens=64):
    prompt = f"<|user|>\n{user_text}</s>\n<|assistant|>\n"
    inputs = llama_tok(prompt, return_tensors="pt").to(llama_model.device)
    with torch.no_grad():
        out = llama_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=llama_tok.eos_token_id,
        )
    return llama_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("Генерация TinyLlama...")
llama_preds = [llama_generate(u) for u in tqdm(users)]
llama_bleu, llama_meteor = compute_metrics(llama_preds, golds)
print(f"TinyLlama (zero-shot): BLEU={llama_bleu:.2f}, METEOR={llama_meteor:.4f}")

# Освобождаем память под T5
del llama_model
torch.cuda.empty_cache()

# ---------- 4. T5-small ----------
print("\n>>> Загрузка t5-small...")
t5_name = "t5-small"
t5_tok = T5Tokenizer.from_pretrained(t5_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_name).to("cuda")
t5_model.eval()

def t5_generate(user_text, max_new_tokens=64):
    inputs = t5_tok(user_text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        out = t5_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        )
    return t5_tok.decode(out[0], skip_special_tokens=True).strip()

print("Генерация t5-small...")
t5_preds = [t5_generate(u) for u in tqdm(users)]
t5_bleu, t5_meteor = compute_metrics(t5_preds, golds)
print(f"t5-small (zero-shot): BLEU={t5_bleu:.2f}, METEOR={t5_meteor:.4f}")

# ---------- 5. Сохраняем результаты ----------
results = {
    "llama": {"bleu": llama_bleu, "meteor": llama_meteor},
    "t5": {"bleu": t5_bleu, "meteor": t5_meteor},
    "n_samples": len(users),
}
with open("metrics_before.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nСохранено в metrics_before.json")
print(json.dumps(results, indent=2, ensure_ascii=False))

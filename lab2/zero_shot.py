import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    T5Tokenizer, T5ForConditionalGeneration,
    BitsAndBytesConfig,
)
from preprocess import load_dailydialog, build_pairs_dataset

# ---------- 1. Данные: берём несколько пар из validation ----------
ds = load_dailydialog()
val_pairs = build_pairs_dataset(ds, "validation")
samples = val_pairs[:3]

print("=" * 70)
print("ЗАПРОСЫ ИЗ VALIDATION (эталонные ответы):")
for i, s in enumerate(samples):
    print(f"[{i}] USER:   {s['user']}")
    print(f"    SYSTEM: {s['system']}")
print("=" * 70)

# ---------- 2. TinyLlama (decoder-only, 4-bit) ----------
print("\n>>> Загрузка TinyLlama-1.1B-Chat (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
llama_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
llama_tok = AutoTokenizer.from_pretrained(llama_name)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
llama_model.eval()
print("TinyLlama загружена.")

def llama_generate(user_text: str, max_new_tokens: int = 64) -> str:
    prompt = f"<|user|>\n{user_text}</s>\n<|assistant|>\n"
    inputs = llama_tok(prompt, return_tensors="pt").to(llama_model.device)
    with torch.no_grad():
        out = llama_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=llama_tok.eos_token_id,
        )
    gen = llama_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()

print("\n" + "=" * 70)
print("TinyLlama ZERO-SHOT:")
for i, s in enumerate(samples):
    gen = llama_generate(s["user"])
    print(f"\n[{i}] USER:   {s['user']}")
    print(f"    GEN:    {gen}")
    print(f"    GOLD:   {s['system']}")
print("=" * 70)

# ---------- 3. T5-small (encoder-decoder) ----------
print("\n>>> Загрузка t5-small...")
t5_name = "t5-small"
t5_tok = T5Tokenizer.from_pretrained(t5_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_name).to("cuda")
t5_model.eval()
print("t5-small загружена.")

def t5_generate(user_text: str, max_new_tokens: int = 64) -> str:
    inputs = t5_tok("reply: " + user_text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        out = t5_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    return t5_tok.decode(out[0], skip_special_tokens=True).strip()

print("\n" + "=" * 70)
print("T5-small ZERO-SHOT:")
for i, s in enumerate(samples):
    gen = t5_generate(s["user"])
    print(f"\n[{i}] USER:   {s['user']}")
    print(f"    GEN:    {gen}")
    print(f"    GOLD:   {s['system']}")
print("=" * 70)

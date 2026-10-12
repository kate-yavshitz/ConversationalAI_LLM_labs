import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------- 0. Настройки ----------
MODEL_NAME = "distilbert-base-uncased"
N_SAMPLES = 15000
MAX_LENGTH = 128
BATCH_SIZE = 32
RANDOM_STATE = 42
CACHE_DIR = "cache_embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device("cpu")


# ---------- 1. Загрузка датасета ----------
dataset = load_dataset("Aldoreni45/sentiment_analysis_preprocessed_dataset")
print(dataset)

texts, labels = [], []
for split in dataset.keys():
    texts.extend(dataset[split]["text"])
    labels.extend(dataset[split]["label"])

texts = texts[:N_SAMPLES]
labels = labels[:N_SAMPLES]

le = LabelEncoder()
labels = le.fit_transform(labels)
print("Классы:", list(le.classes_))

unique, counts = np.unique(labels, return_counts=True)
print("Баланс классов:", dict(zip(unique, counts)))

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)


# ---------- 2. Загрузка модели ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


# ---------- 3. Функции пулинга ----------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0, :]


# ---------- 4. Извлечение эмбеддингов с кэшем ----------
def get_embeddings(texts, pooling, cache_name, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.npy")
    if os.path.exists(cache_path):
        print(f"[cache] Загружаю {cache_path}")
        return np.load(cache_path)

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting ({pooling})"):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            output = model(**encoded)
        if pooling == "mean":
            emb = mean_pooling(output, encoded["attention_mask"])
        else:
            emb = cls_pooling(output)
        all_embeddings.append(emb.cpu().numpy())

    result = np.vstack(all_embeddings)
    np.save(cache_path, result)
    print(f"[cache] Сохранил {cache_path}")
    return result

X_train_mean = get_embeddings(X_train_texts, "mean", "X_train_mean")
X_test_mean  = get_embeddings(X_test_texts,  "mean", "X_test_mean")
X_train_cls  = get_embeddings(X_train_texts, "cls",  "X_train_cls")
X_test_cls   = get_embeddings(X_test_texts,  "cls",  "X_test_cls")

np.save(os.path.join(CACHE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(CACHE_DIR, "y_test.npy"),  y_test)


# ---------- 5. Обучение и метрики ----------
def evaluate(X_train, X_test, y_train, y_test, name):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    metrics = {
        "F1_macro":           f1_score(y_test, y_pred, average="macro"),
        "Precision_macro":    precision_score(y_test, y_pred, average="macro"),
        "Recall_macro":       recall_score(y_test, y_pred, average="macro"),
        "F1_weighted":        f1_score(y_test, y_pred, average="weighted"),
        "Precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "Recall_weighted":    recall_score(y_test, y_pred, average="weighted"),
    }

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_{name.replace(' ', '_')}.png", dpi=120)
    plt.show()

    return metrics

metrics_mean = evaluate(X_train_mean, X_test_mean, y_train, y_test, "Mean Pooling")
metrics_cls  = evaluate(X_train_cls,  X_test_cls,  y_train, y_test, "CLS Pooling")


# ---------- 6. Сравнение ----------
print("\n=== Сравнение стратегий пулинга ===")
print(f"{'Метрика':<22}{'Mean':>10}{'CLS':>10}{'Diff':>10}")
for k in ["F1_macro", "Precision_macro", "Recall_macro",
          "F1_weighted", "Precision_weighted", "Recall_weighted"]:
    diff = metrics_mean[k] - metrics_cls[k]
    print(f"{k:<22}{metrics_mean[k]:>10.4f}{metrics_cls[k]:>10.4f}{diff:>+10.4f}")


# ---------- 7. Автоматический вывод ----------
print("\n=== Вывод ===")
if metrics_mean["F1_macro"] > metrics_cls["F1_macro"]:
    print(f"Mean pooling показал F1(macro) выше на "
          f"{metrics_mean['F1_macro'] - metrics_cls['F1_macro']:+.4f}. "
          f"Это согласуется с теорией: усреднение по всем токенам сохраняет "
          f"больше контекста, чем вектор [CLS] из предобученной модели.")
else:
    print(f"CLS pooling показал F1(macro) выше на "
          f"{metrics_cls['F1_macro'] - metrics_mean['F1_macro']:+.4f}. "
          f"Возможно, для данного датасета доминирующий сигнал тональности "
          f"локализован в начале последовательности.")

gap = abs(metrics_mean["F1_macro"] - metrics_mean["F1_weighted"])
if gap > 0.05:
    print(f"⚠ Разрыв между macro и weighted F1 = {gap:.4f}. "
          f"Классы несбалансированы — интерпретируйте результаты осторожно.")

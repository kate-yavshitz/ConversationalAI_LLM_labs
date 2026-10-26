"""
Нормализация текста DailyDialog и построение пар (user, system).

Используется и при обучении, и при оценке — гарантирует одинаковую обработку.
"""
import re
from datasets import load_dataset

BASE_URL = "https://huggingface.co/datasets/li2017dailydialog/daily_dialog/resolve/refs%2Fconvert%2Fparquet/default/"


def normalize_text(text: str) -> str:
    """
    Приводит реплику к чистому виду:
    1. Склеивает апострофы: "Let ' s" -> "Let's", "don ' t" -> "don't"
    2. Убирает пробелы перед знаками препинания: "fitness ." -> "fitness."
    3. Схлопывает повторные пробелы и обрезает края.
    """
    if not text:
        return ""

    # 1. Апострофы: пробел(ы) вокруг ' -> склеиваем
    #    Примеры: "Let ' s" -> "Let's", "don ' t" -> "don't", "I ' m" -> "I'm"
    text = re.sub(r"\s*'\s*", "'", text)

    # 2. Пробелы перед пунктуацией: "fitness ." -> "fitness."
    #    Знаки: . , ! ? ; : ) ] }
    text = re.sub(r"\s+([.,!?;:\)\]\}])", r"\1", text)

    # 3. Пробел после открывающих скобок/кавычек (на всякий случай)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)

    # 4. Схлопываем все повторные пробелы (включая \t, \n) и обрезаем
    text = " ".join(text.split())

    return text


def load_dailydialog():
    """Загружает DailyDialog из Parquet-файлов (обходит устаревшие скрипты)."""
    return load_dataset(
        "parquet",
        data_files={
            "train": BASE_URL + "train/0000.parquet",
            "validation": BASE_URL + "validation/0000.parquet",
            "test": BASE_URL + "test/0000.parquet",
        },
    )


def dialog_to_pairs(dialog, normalize: bool = True):
    """
    Из списка реплик делает список пар (user, system).
    Чётные индексы (0, 2, 4...) — пользователь.
    Нечётные (1, 3, 5...) — система (ответ).

    Возвращает: [(user_text, system_text), ...]
    """
    pairs = []
    for i in range(1, len(dialog), 2):
        user = dialog[i - 1]
        system = dialog[i]
        if normalize:
            user = normalize_text(user)
            system = normalize_text(system)
        # Пропускаем пустые/слишком короткие пары (бывает в редких случаях)
        if not user or not system:
            continue
        pairs.append((user, system))
    return pairs


def build_pairs_dataset(ds, split: str, normalize: bool = True):
    """
    Превращает HF Dataset (split) в плоский список пар.
    Возвращает список dict: [{"user": ..., "system": ...}, ...]
    """
    flat = []
    for example in ds[split]:
        for user, system in dialog_to_pairs(example["dialog"], normalize=normalize):
            flat.append({"user": user, "system": system})
    return flat


if __name__ == "__main__":
    # Быстрая проверка нормализации
    ds = load_dailydialog()

    print("=== До нормализации ===")
    for utt in ds["train"][0]["dialog"][:6]:
        print(repr(utt))

    print("\n=== После нормализации ===")
    for utt in ds["train"][0]["dialog"][:6]:
        print(repr(normalize_text(utt)))

    print("\n=== Пары (user -> system) из train[0] ===")
    for user, system in dialog_to_pairs(ds["train"][0]["dialog"]):
        print(f"USER:   {user}")
        print(f"SYSTEM: {system}")
        print()

    print("=== Размеры после разворачивания в пары ===")
    for split in ["train", "validation", "test"]:
        pairs = build_pairs_dataset(ds, split)
        print(f"{split}: {len(pairs)} пар")

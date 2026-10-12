# Лабораторная работа №1: Сравнение стратегий пулинга

## Задача
Классификация тональности текста (3 класса: Negative / Neutral / Positive)
на датасете `Aldoreni45/sentiment_analysis_preprocessed_dataset`.

## Модель
`distilbert-base-uncased` (HuggingFace), 15 000 примеров подвыборки,
max_length=128, batch_size=32.

## Проведённые эксперименты

### I. Feature extraction (энкодер заморожен)
- mean pooling + LogisticRegression
- cls pooling + LogisticRegression

### II. Fine-tuning DistilBERT + пулинг поверх дообученной модели
- FT через `ForSequenceClassification`
- mean pooling + LogReg
- cls pooling + LogReg

### III. End-to-end обучение (энкодер + пулинг + голова вместе)
- E2E CLS (стандартный `ForSequenceClassification`)
- E2E Mean (кастомный `DistilBertMeanPoolingClassifier`)

## Результаты (F1 macro)

| Подход | F1_macro | Precision_macro | Recall_macro |
|---|---|---|---|
| Baseline Mean | 0.6195 | 0.6194 | 0.6196 |
| Baseline CLS | 0.6228 | 0.6227 | 0.6229 |
| FT + Mean | 0.6804 | 0.6807 | 0.6804 |
| FT + CLS | 0.6928 | 0.6930 | 0.6928 |
| **E2E Mean** | **0.6976** | **0.6984** | **0.6975** |
| E2E CLS | 0.6861 | 0.6883 | 0.6862 |

## Выводы
1. Fine-tuning критически важен: +0.07 F1 по сравнению с замороженным энкодером.
2. На замороженном энкодере CLS лучше mean — BERT предобучался через [CLS].
3. При end-to-end обучении mean pooling выходит вперёд, т.к. энкодер адаптируется под конкретную стратегию агрегации.
4. Лучший результат — **E2E Mean Pooling**: F1_macro = 0.6976.

## Стек
PyTorch, HuggingFace transformers, datasets, scikit-learn, matplotlib, seaborn.

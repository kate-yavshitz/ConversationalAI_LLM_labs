from datasets import load_dataset

base_url = "https://huggingface.co/datasets/li2017dailydialog/daily_dialog/resolve/refs%2Fconvert%2Fparquet/default/"

ds = load_dataset(
    "parquet",
    data_files={
        "train": base_url + "train/0000.parquet",
        "validation": base_url + "validation/0000.parquet",
        "test": base_url + "test/0000.parquet",
    }
)

print(ds)
print("\n--- Пример диалога (train) ---")
print(ds['train'][0]['dialog'])
print("\n--- Пример диалога (validation) ---")
print(ds['validation'][0]['dialog'])

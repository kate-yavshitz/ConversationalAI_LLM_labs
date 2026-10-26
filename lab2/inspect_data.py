from datasets import load_dataset

base_url = "https://huggingface.co/datasets/li2017dailydialog/daily_dialog/resolve/refs%2Fconvert%2Fparquet/default/"
ds = load_dataset("parquet", data_files={
    "train": base_url + "train/0000.parquet",
    "validation": base_url + "validation/0000.parquet",
    "test": base_url + "test/0000.parquet",
})

print("Размеры сплитов:", {k: len(v) for k, v in ds.items()})
print("Колонки:", ds['train'].column_names)
print("\nПервый пример целиком:")
print(ds['train'][0])
print("\nПять первых реплик train[0]:")
for i, utt in enumerate(ds['train'][0]['dialog']):
    print(f"  {i}: {utt}")

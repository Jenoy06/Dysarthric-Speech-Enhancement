import pandas as pd
from pathlib import Path

CSV_PATH = Path(
    r"D:\Dysarthric Speech Enhancement\torgo_inventory.csv"
)

df = pd.read_csv(CSV_PATH)

print("=" * 50)
print("TORGO INVENTORY VERIFICATION")
print("=" * 50)

print("\n1. Total inventory rows:")
print(len(df))

print("\n2. Missing values:")
print(df.isnull().sum())

print("\n3. Recordings by group:")
print(df["group"].value_counts().sort_index())

print("\n4. Recordings by speaker:")
print(
    df.groupby(["group", "speaker_id"])
      .size()
      .sort_index()
      .to_string()
)

print("\n5. Recordings by microphone and group:")
print(
    pd.crosstab(df["group"], df["microphone"])
)

print("\n6. Duplicate file paths:")
print(df["relative_path"].duplicated().sum())

print("\n7. Check whether WAV files exist:")
dataset_root = Path(
    r"D:\Dysarthric Speech Enhancement\dataset"
)

sample = df.sample(min(10, len(df)), random_state=42)

for _, row in sample.iterrows():
    path = dataset_root / row["relative_path"]
    print(
        "OK" if path.is_file() else "MISSING",
        row["relative_path"]
    )

print("\nVerification finished.")
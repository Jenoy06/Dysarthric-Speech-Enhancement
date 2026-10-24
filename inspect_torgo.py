
from pathlib import Path
from collections import Counter

# Path to your TORGO dataset folder
DATASET_PATH = Path(r"D:\Dysarthric Speech Enhancement\dataset")

if not DATASET_PATH.exists():
    print("Folder not found:", DATASET_PATH)
else:
    files = [p for p in DATASET_PATH.rglob("*") if p.is_file()]

    print("Dataset folder:", DATASET_PATH)
    print("Total files:", len(files))

    # Count file types
    extensions = Counter(
        p.suffix.lower() or "[no extension]"
        for p in files
    )

    print("\nFile types:")
    for extension, count in extensions.most_common():
        print(f"{extension}: {count}")

    # Count audio files
    wav_files = [
        p for p in files
        if p.suffix.lower() == ".wav"
    ]

    print("\nTotal WAV recordings:", len(wav_files))

    print("\nFirst 30 WAV file paths:")
    for wav in wav_files[:30]:
        print(wav.relative_to(DATASET_PATH))

    # Find possible metadata and documentation files
    print("\nPossible metadata/documentation files:")

    for file in files:
        if (
            file.suffix.lower()
            in {".csv", ".tsv", ".json", ".txt", ".xlsx"}
            or file.name.lower().startswith("readme")
        ):
            print(file.relative_to(DATASET_PATH))

from pathlib import Path
from collections import Counter

DATASET_PATH = Path(
    r"D:\Dysarthric Speech Enhancement\dataset"
)

# Find all WAV recordings
wav_files = list(DATASET_PATH.rglob("*.wav"))

print("=" * 60)
print("TORGO SPEAKER AND FOLDER INSPECTION")
print("=" * 60)

print("\nTotal WAV recordings:", len(wav_files))

# Count recordings by top-level group inside the dataset
group_counts = Counter()

for wav in wav_files:
    relative_path = wav.relative_to(DATASET_PATH)
    parts = relative_path.parts

    # Example path:
    # torgo-audio/versions/1/M_Dys/...
    if len(parts) >= 4:
        group = parts[3]
    else:
        group = "Unknown"

    group_counts[group] += 1

print("\nRecordings by group:")
for group, count in group_counts.most_common():
    print(f"{group}: {count}")

# Identify folders that directly contain WAV recordings
recording_folders = {
    wav.parent for wav in wav_files
}

print("\nFolders containing WAV recordings:")
for folder in sorted(recording_folders):
    print(folder.relative_to(DATASET_PATH))

print("\nTotal folders containing WAV recordings:",
      len(recording_folders))

# Count recordings in each recording folder
print("\nRecording counts per folder:")
for folder in sorted(recording_folders):
    count = sum(1 for wav in wav_files if wav.parent == folder)
    print(f"{folder.relative_to(DATASET_PATH)}: {count}")
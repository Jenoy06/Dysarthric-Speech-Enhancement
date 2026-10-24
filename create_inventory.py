from pathlib import Path
import csv
import re
from collections import Counter

DATASET_PATH = Path(
    r"D:\Dysarthric Speech Enhancement\dataset"
)

OUTPUT_CSV = Path(
    r"D:\Dysarthric Speech Enhancement\torgo_inventory.csv"
)

wav_files = list(DATASET_PATH.rglob("*.wav"))

rows = []
group_counts = Counter()
microphone_counts = Counter()
speaker_ids = set()

for wav in wav_files:
    relative = wav.relative_to(DATASET_PATH)
    parts = relative.parts

    # Expected structure:
    # torgo-audio/versions/1/M_Dys/wav_arrayMic_M01S01/file.wav
    if len(parts) < 6:
        print("Unexpected path structure:", relative)
        continue

    group = parts[3]
    folder_name = wav.parent.name

    # Extract speaker/session code from the folder name.
    # Examples: wav_arrayMic_M01S01, wav_arrayMic_FC02S03
    match = re.search(
        r"wav_(?:arrayMic|headMic)_(.+)",
        folder_name,
        re.IGNORECASE
    )

    code = match.group(1) if match else "UNKNOWN"

    # Remove the final session marker, when present.
    # Example: M01S01 -> M01
    speaker_id = re.sub(r"S\d+$", "", code)

    if speaker_id != "UNKNOWN":
        speaker_ids.add(speaker_id)

    if "arrayMic" in folder_name:
        microphone = "arrayMic"
    elif "headMic" in folder_name:
        microphone = "headMic"
    else:
        microphone = "unknown"

    if group not in {"M_Con", "F_Con", "M_Dys", "F_Dys"}:
        print("Unrecognized group:", group, relative)

    rows.append({
        "relative_path": str(relative),
        "group": group,
        "speaker_id": speaker_id,
        "folder_code": code,
        "microphone": microphone,
        "filename": wav.name
    })

    group_counts[group] += 1
    microphone_counts[microphone] += 1

# Save the inventory as a CSV file.
with OUTPUT_CSV.open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "relative_path",
            "group",
            "speaker_id",
            "folder_code",
            "microphone",
            "filename"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nINVENTORY COMPLETE")
print("WAV files found:", len(wav_files))
print("Rows written:", len(rows))
print("Output CSV:", OUTPUT_CSV)

print("\nRecordings by group:")
for group, count in sorted(group_counts.items()):
    print(f"{group}: {count}")

print("\nRecordings by microphone:")
for mic, count in sorted(microphone_counts.items()):
    print(f"{mic}: {count}")

print("\nUnique speaker IDs detected:", len(speaker_ids))
print("Speaker IDs:", ", ".join(sorted(speaker_ids)))
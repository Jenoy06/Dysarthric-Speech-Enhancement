from pathlib import Path
import pandas as pd
import soundfile as sf

# Main project folder
BASE_DIR = Path(r"D:\Dysarthric Speech Enhancement")

# Dataset audio folder
PROJECT_DIR = BASE_DIR / "dataset"

# Inventory and report are stored in the main project folder
CSV_PATH = BASE_DIR / "torgo_inventory.csv"
REPORT_PATH = BASE_DIR / "audio_inspection_report.csv"

print("=" * 55)
print("TORGO AUDIO INSPECTION")
print("=" * 55)

# Load inventory
df = pd.read_csv(CSV_PATH)

required_columns = {"relative_path", "group", "speaker_id"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing inventory columns: {missing_columns}")

results = []
total = len(df)

for index, row in df.iterrows():
    relative_path = Path(str(row["relative_path"]))
    audio_path = PROJECT_DIR / relative_path

    result = {
        "relative_path": str(relative_path),
        "group": row["group"],
        "speaker_id": row["speaker_id"],
        "exists": audio_path.is_file(),
        "readable": False,
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
        "error": ""
    }

    if not audio_path.is_file():
        result["error"] = "File not found"
    else:
        try:
            info = sf.info(str(audio_path))

            result["readable"] = True
            result["duration_seconds"] = info.duration
            result["sample_rate"] = info.samplerate
            result["channels"] = info.channels

            if info.frames == 0:
                result["error"] = "Empty audio file"
            elif info.duration <= 0:
                result["error"] = "Invalid duration"

        except Exception as exc:
            result["error"] = str(exc)

    results.append(result)

    if (index + 1) % 500 == 0 or index + 1 == total:
        print(f"Checked {index + 1}/{total} recordings")

# Save detailed report
report = pd.DataFrame(results)
report.to_csv(REPORT_PATH, index=False)

# Summary
print("\n" + "=" * 55)
print("INSPECTION SUMMARY")
print("=" * 55)

print("Inventory rows:", total)
print("Existing files:", int(report["exists"].sum()))
print("Readable audio files:", int(report["readable"].sum()))
print("Missing files:", int((~report["exists"]).sum()))
print("Unreadable files:", int(
    (report["exists"] & ~report["readable"]).sum()
))

valid_audio = report[
    report["readable"]
    & report["duration_seconds"].notna()
    & (report["duration_seconds"] > 0)
]

print("Audio with positive duration:", len(valid_audio))

print("\nSample rates:")
print(report.loc[report["readable"], "sample_rate"].value_counts())

print("\nChannel counts:")
print(report.loc[report["readable"], "channels"].value_counts())

print("\nGroups:")
print(report.groupby("group")["readable"].sum())

print("\nReport saved to:")
print(REPORT_PATH)

print("\nInspection finished.")


from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"D:\Dysarthric Speech Enhancement")
REPORT_PATH = BASE_DIR / "audio_inspection_report.csv"

df = pd.read_csv(REPORT_PATH)

# Find files that could not be read
unreadable = df[~df["readable"]]

print("=" * 55)
print("UNREADABLE AUDIO FILES")
print("=" * 55)

if unreadable.empty:
    print("No unreadable files found.")
else:
    print(
        unreadable[
            ["relative_path", "group", "speaker_id", "error"]
        ].to_string(index=False)
    )

# Find files with missing, zero, or negative duration
duration = pd.to_numeric(
    df["duration_seconds"], errors="coerce"
)

invalid_duration = df[
    duration.isna() | (duration <= 0)
]

print("\n" + "=" * 55)
print("FILES WITH INVALID OR MISSING DURATION")
print("=" * 55)

if invalid_duration.empty:
    print("No invalid durations found.")
else:
    print(
        invalid_duration[
            [
                "relative_path",
                "group",
                "speaker_id",
                "readable",
                "duration_seconds",
                "error"
            ]
        ].to_string(index=False)
    )

print("\n" + "=" * 55)
print("CHECK COMPLETE")
print("=" * 55)

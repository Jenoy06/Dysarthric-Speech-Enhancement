
from pathlib import Path
import struct

BASE_DIR = Path(r"D:\Dysarthric Speech Enhancement")
DATASET_DIR = BASE_DIR / "dataset"

bad_files = [
    Path(
        r"torgo-audio\versions\1\F_Dys\wav_headMic_F01"
        r"\wav_headMic_F01_0067.wav"
    ),
    Path(
        r"torgo-audio\versions\1\F_Dys\wav_headMic_F01"
        r"\wav_headMic_F01_0068.wav"
    ),
    Path(
        r"torgo-audio\versions\1\F_Con\wav_arrayMic_FC01S01"
        r"\wav_arrayMic_FC01S01_0256.wav"
    ),
]

for relative_path in bad_files:
    path = DATASET_DIR / relative_path

    print("\n" + "=" * 60)
    print("File:", path.name)
    print("Exists:", path.is_file())

    if not path.is_file():
        continue

    print("Size in bytes:", path.stat().st_size)

    with path.open("rb") as f:
        header = f.read(64)

    print("First 32 bytes:", header[:32])
    print("Header as text:", repr(header[:32]))

    if len(header) >= 12:
        print("First four bytes:", repr(header[:4]))
        print("Bytes 8-11:", repr(header[8:12]))

        if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            print("Header resembles a standard WAV file.")
        elif header[:4] == b"RF64" and header[8:12] == b"WAVE":
            print("Header resembles an RF64 audio file.")
        else:
            print("Header does not match standard RIFF/WAVE.")

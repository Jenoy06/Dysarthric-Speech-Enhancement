import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_structured_dataset():
    base_dir = "images"
    
    # Define sub-directory paths for Dysarthric
    dys_base = os.path.join(base_dir, "dysarthric")
    dys_input_dir = os.path.join(dys_base, "input")
    dys_mel_dir = os.path.join(dys_base, "mel")
    dys_mfcc_dir = os.path.join(dys_base, "mfcc")

    # Define sub-directory paths for Controlled
    ctrl_base = os.path.join(base_dir, "controlled")
    ctrl_input_dir = os.path.join(ctrl_base, "input")
    ctrl_mel_dir = os.path.join(ctrl_base, "mel")
    ctrl_mfcc_dir = os.path.join(ctrl_base, "mfcc")

    # Define DTW Aligned directory
    dtw_dir = os.path.join(base_dir, "dtw_aligned")

    # Create all directories
    for path in [dys_input_dir, dys_mel_dir, dys_mfcc_dir, 
                 ctrl_input_dir, ctrl_mel_dir, ctrl_mfcc_dir, dtw_dir]:
        os.makedirs(path, exist_ok=True)

    print("Created structured nested directories under 'images/'.")

    # Search for files
    dysarthric_files = sorted(glob.glob(r"dataset/torgo-audio/versions/1/**/*Dys*/**/*.wav", recursive=True))
    controlled_files = sorted(glob.glob(r"dataset/torgo-audio/versions/1/**/*Con*/**/*.wav", recursive=True))

    print(f"Found {len(dysarthric_files)} Dysarthric files.")
    print(f"Found {len(controlled_files)} Controlled files.")

    if len(dysarthric_files) < 25 or len(controlled_files) < 25:
        print("\n[!] WARNING: Could not find at least 25 files for both categories.")
        return

    # Select 25 files each
    dys_selected = dysarthric_files[:25]
    ctrl_selected = controlled_files[:25]
    
    num_samples = len(dys_selected)
    print(f"\nProcessing {num_samples} paired samples...")

    for i in range(num_samples):
        dys_path = dys_selected[i]
        ctrl_path = ctrl_selected[i]
        
        dys_name = os.path.splitext(os.path.basename(dys_path))[0]
        ctrl_name = os.path.splitext(os.path.basename(ctrl_path))[0]
        
        print(f"[{i+1}/{num_samples}] Pair -> Dys: {dys_name} | Ctrl: {ctrl_name}")

        try:
            # Load Audio
            y_dys, sr_dys = librosa.load(dys_path, sr=None)
            y_ctrl, sr_ctrl = librosa.load(ctrl_path, sr=None)

            # ==========================================
            # DYSARTHRIC FEATURES (Saved in subfolders)
            # ==========================================
            # 1. Input Waveform
            plt.figure(figsize=(10, 3))
            librosa.display.waveshow(y_dys, sr=sr_dys, alpha=0.8)
            plt.title(f"Dysarthric Input Waveform - {dys_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(dys_input_dir, f"{dys_name}_waveform.png"), dpi=150)
            plt.close()

            # 2. Log Mel Spectrogram
            mel_dys = librosa.feature.melspectrogram(y=y_dys, sr=sr_dys, n_mels=128)
            mel_dys_db = librosa.power_to_db(mel_dys, ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_dys_db, sr=sr_dys, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Dysarthric Log Mel Spectrogram - {dys_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(dys_mel_dir, f"{dys_name}_log_mel.png"), dpi=150)
            plt.close()

            # 3. MFCC
            mfcc_dys = librosa.feature.mfcc(y=y_dys, sr=sr_dys, n_mfcc=13)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfcc_dys, sr=sr_dys, x_axis='time')
            plt.colorbar()
            plt.title(f"Dysarthric MFCC - {dys_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(dys_mfcc_dir, f"{dys_name}_mfcc.png"), dpi=150)
            plt.close()

            # ==========================================
            # CONTROLLED FEATURES (Saved in subfolders)
            # ==========================================
            # 1. Input Waveform
            plt.figure(figsize=(10, 3))
            librosa.display.waveshow(y_ctrl, sr=sr_ctrl, alpha=0.8, color='green')
            plt.title(f"Controlled Input Waveform - {ctrl_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(ctrl_input_dir, f"{ctrl_name}_waveform.png"), dpi=150)
            plt.close()

            # 2. Log Mel Spectrogram
            mel_ctrl = librosa.feature.melspectrogram(y=y_ctrl, sr=sr_ctrl, n_mels=128)
            mel_ctrl_db = librosa.power_to_db(mel_ctrl, ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_ctrl_db, sr=sr_ctrl, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Controlled Log Mel Spectrogram - {ctrl_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(ctrl_mel_dir, f"{ctrl_name}_log_mel.png"), dpi=150)
            plt.close()

            # 3. MFCC
            mfcc_ctrl = librosa.feature.mfcc(y=y_ctrl, sr=sr_ctrl, n_mfcc=13)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfcc_ctrl, sr=sr_ctrl, x_axis='time')
            plt.colorbar()
            plt.title(f"Controlled MFCC - {ctrl_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(ctrl_mfcc_dir, f"{ctrl_name}_mfcc.png"), dpi=150)
            plt.close()

            # ==========================================
            # CROSS-DTW ALIGNED SPECTROGRAM
            # ==========================================
            _, wp = librosa.sequence.dtw(X=mfcc_dys, Y=mfcc_ctrl, metric='cosine')
            warp_indices = wp[:, 1][::-1]
            aligned_mel_db = mel_ctrl_db[:, warp_indices]

            plt.figure(figsize=(10, 5))
            plt.subplot(2, 1, 1)
            librosa.display.specshow(mel_dys_db, sr=sr_dys, x_axis='time', y_axis='mel')
            plt.title(f"DTW Source (Dysarthric): {dys_name}")
            plt.colorbar(format='%+2.0f dB')

            plt.subplot(2, 1, 2)
            librosa.display.specshow(aligned_mel_db, sr=sr_dys, x_axis='time', y_axis='mel')
            plt.title(f"DTW Aligned Target (Controlled warped): {ctrl_name}")
            plt.colorbar(format='%+2.0f dB')

            plt.tight_layout()
            dtw_output_name = f"pair_{i+1:02d}_{dys_name}_to_{ctrl_name}_dtw.png"
            plt.savefig(os.path.join(dtw_dir, dtw_output_name), dpi=150)
            plt.close()

        except Exception as e:
            print(f"Error processing pair {dys_name} & {ctrl_name}: {e}")

    print("\nProcessing complete! Directory layout created successfully:")
    print("images/")
    print(" ├── dysarthric/ (input/, mel/, mfcc/)")
    print(" ├── controlled/ (input/, mel/, mfcc/)")
    print(" └── dtw_aligned/")

if __name__ == "__main__":
    generate_structured_dataset()
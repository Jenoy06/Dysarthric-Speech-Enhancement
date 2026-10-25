import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm

def generate_sample_visualizations():
    csv_file = 'processed_dataset.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found! Run your preprocessing script first.")
        return

    df = pd.read_csv(csv_file)
    print("CSV Columns found:", list(df.columns))

    # Identify speaker group column dynamically
    group_col = 'group' if 'group' in df.columns else ('speaker_id' if 'speaker_id' in df.columns else df.columns[0])

    # Helper function to check if a row represents a control speaker
    def check_is_control(row):
        val_str = str(row[group_col]).lower()
        spk_str = str(row.get('speaker_id', '')).lower()
        # TORGO control speakers usually contain 'control', 'fc', or 'mc' (e.g. FC01, MC01, C)
        is_ctrl = any(keyword in val_str or keyword in spk_str for keyword in ['control', 'fc', 'mc', '_c', 'c0'])
        return is_ctrl

    df['is_control'] = df.apply(check_is_control, axis=1)

    control_df = df[df['is_control'] == True].head(50)
    dys_df = df[df['is_control'] == False].head(50)

    print(f"Found {len(control_df)} Control samples and {len(dys_df)} Dysarthric samples.")

    sample_df = pd.concat([control_df, dys_df])

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Saving Visualizations"):
        speaker_id = str(row.get('speaker_id', 'spk'))
        filename = str(row.get('filename', f'sample_{idx}')).replace('.wav', '')
        
        mel_path = row.get('log_mel_path', '')
        mfcc_path = row.get('mfcc_path', '')

        if not os.path.exists(mel_path) or not os.path.exists(mfcc_path):
            continue

        category = 'control' if row['is_control'] else 'dysarthric'

        # Output subdirectories
        mel_out_dir = os.path.join('visualizations', category, 'log_mel')
        mfcc_out_dir = os.path.join('visualizations', category, 'mfcc')
        os.makedirs(mel_out_dir, exist_ok=True)
        os.makedirs(mfcc_out_dir, exist_ok=True)

        # Load and render features
        mel_spec = np.load(mel_path)
        mfcc = np.load(mfcc_path)

        # --- Log-Mel Spectrogram ---
        plt.figure(figsize=(8, 3))
        librosa.display.specshow(mel_spec, sr=16000, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Log-Mel | {category.upper()} | {speaker_id}_{filename}')
        plt.tight_layout()
        plt.savefig(os.path.join(mel_out_dir, f"{speaker_id}_{filename}_mel.png"), dpi=100)
        plt.close()

        # --- MFCC ---
        plt.figure(figsize=(8, 3))
        librosa.display.specshow(mfcc, sr=16000, hop_length=512, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'MFCC | {category.upper()} | {speaker_id}_{filename}')
        plt.tight_layout()
        plt.savefig(os.path.join(mfcc_out_dir, f"{speaker_id}_{filename}_mfcc.png"), dpi=100)
        plt.close()

    print("\nProcessing complete! Check the updated 'visualizations' directory in VS Code.")

if __name__ == '__main__':
    generate_sample_visualizations()
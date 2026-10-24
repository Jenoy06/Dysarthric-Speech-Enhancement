import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

CSV_PATH = 'torgo_inventory.csv'
OUTPUT_DIR = 'processed_features'
SAMPLE_RATE = 16000
DURATION = 3.0
MAX_LEN = int(SAMPLE_RATE * DURATION)
N_MFCC = 40
N_MELS = 80  # Matches your friend's CNN model shape [1, 1, 80, 200]

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(y) < MAX_LEN:
            y = np.pad(y, (0, MAX_LEN - len(y)), mode='constant')
        else:
            y = y[:MAX_LEN]
            
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=512)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        if log_mel_spec.shape[1] < 200:
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 200 - log_mel_spec.shape[1])), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :200]

        return mfcc, log_mel_spec, True
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, False

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} does not exist.")

    os.makedirs(os.path.join(OUTPUT_DIR, 'mfcc'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'log_mel'), exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} total audio records from {CSV_PATH}.")

    processed_data = []
    success_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        rel_path = str(row['relative_path'])
        
        # Check possible locations where the dataset might be stored locally
        possible_paths = [
            rel_path,
            os.path.join('dataset', rel_path),
            os.path.join('.', rel_path),
            rel_path.replace('torgo-audio', 'dataset/torgo-audio')
        ]
        
        file_path = None
        for p in possible_paths:
            if os.path.exists(p):
                file_path = p
                break

        if not file_path:
            continue
            
        mfcc, log_mel, success = extract_audio_features(file_path)
        
        if success:
            file_id = f"sample_{idx:05d}"
            mfcc_path = os.path.join(OUTPUT_DIR, 'mfcc', f"{file_id}_mfcc.npy")
            mel_path = os.path.join(OUTPUT_DIR, 'log_mel', f"{file_id}_mel.npy")
            
            np.save(mfcc_path, mfcc)
            np.save(mel_path, log_mel)
            
            row_dict = row.to_dict()
            row_dict['resolved_path'] = file_path
            row_dict['mfcc_path'] = mfcc_path
            row_dict['log_mel_path'] = mel_path
            processed_data.append(row_dict)
            success_count += 1

    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv('processed_dataset.csv', index=False)
    print(f"\nSuccessfully processed and saved {success_count} feature vectors!")

if __name__ == '__main__':
    main()
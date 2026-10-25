import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def extract_log_mel_spectrogram(audio_path, sr=16000, n_mels=80, n_fft=1024, hop_length=256):
    y, _ = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    return librosa.power_to_db(mel_spec, ref=np.max)

def align_spectrograms_with_dtw(dysarthric_spec, control_spec):
    D, wp = librosa.sequence.dtw(X=dysarthric_spec, Y=control_spec, metric='euclidean')
    aligned_frames = []
    for ctrl_idx in range(control_spec.shape[1]):
        matching_dys_indices = wp[wp[:, 1] == ctrl_idx, 0]
        if len(matching_dys_indices) > 0:
            mean_frame = np.mean(dysarthric_spec[:, matching_dys_indices], axis=1)
        else:
            mean_frame = dysarthric_spec[:, min(ctrl_idx, dysarthric_spec.shape[1]-1)]
        aligned_frames.append(mean_frame)
    return np.array(aligned_frames).T

def main():
    viz_dir = "visualizations"
    output_dir = os.path.join(viz_dir, "dtw_aligned")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Map all dataset .wav files by filename and group by utterance suffix (e.g., '_0001.wav')
    all_wavs = glob.glob("dataset/**/*.wav", recursive=True)
    wav_map = {os.path.basename(w).replace(".wav", ""): w for w in all_wavs}
    print(f"📦 Total .wav files mapped from dataset: {len(wav_map)}")

    # 2. Scan existing visualizations for the 100 audio IDs
    all_viz_files = glob.glob(os.path.join(viz_dir, "**/*.*"), recursive=True)
    audio_ids = set()
    for vf in all_viz_files:
        filename = os.path.basename(vf)
        for suffix in ["_mel.png", "_mfcc.png", "_spectrogram.png", ".png"]:
            if filename.endswith(suffix):
                audio_ids.add(filename[:-len(suffix)])
                break
                
    print(f"📊 Found {len(audio_ids)} unique audio IDs from visualizations.")
    
    matched_count = 0
    
    # 3. Match each visualization ID with a control reference sharing the same prompt suffix
    for dys_id in sorted(list(audio_ids)):
        dys_path = wav_map.get(dys_id)
        if not dys_path:
            continue
            
        # Extract utterance suffix (e.g., '_0001') from the filename
        # TORGO filenames typically end with an utterance number like _0001, _0002, etc.
        suffix_key = dys_id.split('_')[-1] 
        
        # Look for a control audio file that shares this suffix
        control_path = None
        for name, path in wav_map.items(('Con' in path or 'control' in path.lower()) for path in [path]): # search control speakers
            pass
            
        # Simple fallback: find any wav file with the same ending digits in a control folder
        for name, path in wav_map.items():
            if name.endswith(suffix_key) and ("Con" in path or "control" in path.lower() or "FC" in path or "MC" in path):
                if path != dys_path:
                    control_path = path
                    break
                    
        # If no specific control speaker match found, pick the first available different file as fallback
        if not control_path:
            for name, path in wav_map.items():
                if path != dys_path:
                    control_path = path
                    break

        if dys_path and control_path:
            try:
                dys_spec = extract_log_mel_spectrogram(dys_path)
                ctrl_spec = extract_log_mel_spectrogram(control_path)
                aligned_spec = align_spectrograms_with_dtw(dys_spec, ctrl_spec)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                img = librosa.display.specshow(aligned_spec, x_axis='time', y_axis='mel', sr=16000, ax=ax)
                ax.set_title(f"DTW Aligned Spectrogram: {dys_id}")
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                plt.tight_layout()
                
                save_path = os.path.join(output_dir, f"{dys_id}_dtw_aligned.png")
                plt.savefig(save_path)
                plt.close(fig)
                matched_count += 1
                print(f"[{matched_count}] Aligned & saved: {dys_id}")
            except Exception as e:
                print(f"❌ Error on {dys_id}: {e}")
                
    print(f"\n✅ Finished! Successfully saved {matched_count} DTW-aligned visual representations in '{output_dir}'.")

if __name__ == "__main__":
    main()
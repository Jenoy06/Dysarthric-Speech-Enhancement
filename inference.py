import os
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from enhancement_model import SpeechEnhancementCNN

def generate_enhanced_audio():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 1. Load the trained model weights
    model = SpeechEnhancementCNN().to(device)
    model_path = 'saved_models/speech_enhancement_cnn.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Pick a sample from your dataset manifest
    df = pd.read_csv('processed_dataset.csv')
    sample = df.iloc[0] # Picks the first sample in your CSV
    
    mel_path = sample['log_mel_path']
    speaker_id = sample['speaker_id']
    filename = sample['filename']
    
    print(f"Running inference on Speaker: {speaker_id}, File: {filename}")

    # 3. Load the Log-Mel tensor
    mel_spec = np.load(mel_path)
    
    # PyTorch expects shape: (Batch, Channel, Height, Width) -> (1, 1, n_mels, time)
    input_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 4. Forward pass through the trained U-Net model
    with torch.no_grad():
        output_tensor = model(input_tensor)

    enhanced_mel = output_tensor.squeeze().cpu().numpy()

    # 5. Convert Log-Mel Spectrogram back to Audio Waveform
    mel_power = librosa.db_to_power(enhanced_mel)
    stft_mag = librosa.feature.inverse.mel_to_stft(mel_power, sr=16000, n_fft=2048, power=2.0)
    audio_waveform = librosa.griffinlim(stft_mag, hop_length=512)

    # 6. Save the enhanced audio file
    os.makedirs('enhanced_output', exist_ok=True)
    output_wav_path = os.path.join('enhanced_output', f"enhanced_{speaker_id}_{filename}")
    sf.write(output_wav_path, audio_waveform, 16000)

    print(f"Success! Enhanced audio saved to: {output_wav_path}")

if __name__ == '__main__':
    generate_enhanced_audio()
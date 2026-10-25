import os
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from enhancement_model import SpeechEnhancementCNN

def enhance_specific_file():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load trained model weights
    model = SpeechEnhancementCNN().to(device)
    model_path = 'saved_models/speech_enhancement_cnn.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Load dataset manifest
    csv_path = 'processed_dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
        
    df = pd.read_csv(csv_path)
    
    # 3. Target your exact speaker and file by combining keywords
    speaker_id = "F01"
    file_id = "0007"
    
    # This filters rows that contain BOTH 'F01' and '0007' in the file path
    filtered_df = df[df['log_mel_path'].str.contains(speaker_id) & df['log_mel_path'].str.contains(file_id)]

    if len(filtered_df) == 0:
        print(f"Error: Could not find a file matching speaker '{speaker_id}' and file '{file_id}'.")
        return

    selected_row = filtered_df.iloc[0]
    mel_path = selected_row['log_mel_path']
    print(f"Successfully targeted exact file path: {mel_path}")
    
    # 4. Load the Log-Mel spectrogram
    if not os.path.exists(mel_path):
        print(f"Error: Spectrogram file not found at {mel_path}")
        return
        
    mel_spec = np.load(mel_path)
    input_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 5. Pass through model
    with torch.no_grad():
        enhanced_tensor = model(input_tensor)
        
    enhanced_mel = enhanced_tensor.squeeze().cpu().numpy()
    
    # 6. Convert Log-Mel back to Audio Waveform using Griffin-Lim
    print("Converting spectrogram back to audio...")
    reconstructed_audio = librosa.feature.inverse.mel_to_audio(
        enhanced_mel, 
        sr=16000, 
        n_fft=400, 
        hop_length=160, 
        win_length=400
    )
    
    # 7. Save the specific enhanced audio file
    os.makedirs('enhanced_output', exist_ok=True)
    output_path = f'enhanced_output/enhanced_{speaker_id}_{file_id}.wav'
    sf.write(output_path, reconstructed_audio, 16000)
    
    print(f"Success! Enhanced audio saved to: {output_path}")

if __name__ == '__main__':
    enhance_specific_file()
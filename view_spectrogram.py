import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Load one of your saved Log-Mel spectrogram files
file_path = "processed_features/log_mel/sample_00000_mel.npy"
mel_spec = np.load(file_path)

print(f"Loaded spectrogram shape: {mel_spec.shape}")

# Plot and display the spectrogram as an image
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec, sr=16000, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Mel Spectrogram Visualization (Sample 0)')
plt.tight_layout()
plt.show()

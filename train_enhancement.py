import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train_cnn import TORGOMelDataset  # Uses your existing dataset loader
from enhancement_model import SpeechEnhancementCNN
from dtw_alignment import extract_log_mel_spectrogram, align_spectrograms_with_dtw

# 1. Load your paired audio file paths from TORGO
dysarthric_path = r"D:\Dysarthric Speech Enhancement\torgo\dysarthric_sample.wav"
control_path = r"D:\Dysarthric Speech Enhancement\torgo\control_sample.wav"

# 2. Extract Log-Mel spectrograms
dys_spec = extract_log_mel_spectrogram(dysarthric_path)
ctrl_spec = extract_log_mel_spectrogram(control_path)

# 3. Apply DTW Alignment so shapes match perfectly for your neural network
aligned_dys_spec = align_spectrograms_with_dtw(dys_spec, ctrl_spec)

print(f"Original Dysarthric Shape: {dys_spec.shape}")
print(f"Target Control Shape:    {ctrl_spec.shape}")
print(f"DTW Aligned Shape:       {aligned_dys_spec.shape}") 
def train_speech_enhancement():
    # 1. Setup device (using CPU as configured for your system)
    # Automatically use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if device.type == 'cuda':
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"Training on device: {device}")

    # 2. Load Dataset & DataLoader
    dataset = TORGOMelDataset('processed_dataset.csv')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # 3. Initialize Model, Loss Function, and Optimizer
    model = SpeechEnhancementCNN().to(device)
    
    # L1 Loss (Mean Absolute Error) works exceptionally well for spectrogram reconstruction
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    num_epochs = 3  # Start with a quick test run of 3 epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass through your U-Net CNN
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass & optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Finished | Average Loss: {epoch_loss:.4f} ===")

    # 5. Save the trained weights
    os.makedirs('saved_models', exist_ok=True)
    model_path = 'saved_models/speech_enhancement_cnn.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Training complete! Model weights saved to '{model_path}'.")

if __name__ == '__main__':
    train_speech_enhancement()
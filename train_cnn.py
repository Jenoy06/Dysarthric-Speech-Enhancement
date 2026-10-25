import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from enhancement_model import SpeechEnhancementCNN

# 1. The Denoising Dataset Class (Self-Supervised)
class SelfSupervisedSpeechDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel_path = row['log_mel_path']
        
        # Load the spectrogram (shape: n_mels, time)
        mel_spec = np.load(mel_path)
        
        # Convert to PyTorch tensor -> shape: (1, n_mels, time)
        target_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        
        # Add random noise to create the input (Denoising autoencoder trick)
        noise = torch.randn_like(target_tensor) * 0.2  
        input_tensor = target_tensor + noise
        
        return input_tensor, target_tensor

def train_model():
    # Setup device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # 2. Load Dataset
    csv_path = 'processed_dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found! Please check your file paths.")
        return

    dataset = SelfSupervisedSpeechDataset(csv_path)
    # Using batch_size=4 to fit nicely in your RTX 4050 memory
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # 3. Initialize Model, Loss, and Optimizer
    model = SpeechEnhancementCNN().to(device)
    criterion = nn.L1Loss()  # L1 Loss for sharp spectrogram reconstruction
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop (Let's run 5 epochs to start)
    num_epochs = 5
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass & optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Finished | Average Loss: {avg_loss:.4f} ===")

    # 5. Save Model Weights
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'saved_models/speech_enhancement_cnn.pth')
    print("Training complete! Model weights saved to 'saved_models/speech_enhancement_cnn.pth'.")

if __name__ == '__main__':
    train_model()
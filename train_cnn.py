import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from enhancement_model import SpeechEnhancementCNN

# 1. Define Dataset class to load saved Log-Mel spectrograms
class TORGOMelDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel_path = row['log_mel_path']
        
        # Load the saved numpy array [80, 200]
        mel_spec = np.load(mel_path)
        
        # Add channel dimension to make it [1, 80, 200] for PyTorch CNN
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        
        # For autoencoder / enhancement, input and target can be the same (or clean vs noisy pairs)
        return mel_tensor, mel_tensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset manifest
    dataset = TORGOMelDataset('processed_dataset.csv')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize model
    model = SpeechEnhancementCNN().to(device)
    print("Initialized SpeechEnhancementCNN model successfully!")

    # Quick test batch run
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            
        print(f"Batch {batch_idx}: Input shape {inputs.shape} -> Output shape {outputs.shape}")
        break  # Test just the first batch

if __name__ == '__main__':
    main()
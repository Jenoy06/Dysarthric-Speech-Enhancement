import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        return e1, e2, e3

class SpeechBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = ConvBlock(128, 256)

    def forward(self, x):
        return self.bottleneck(x)

class SpeechDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)
        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, b, e1, e2, e3):
        # Decoder stage 1
        d3 = F.interpolate(
            b,
            size=e3.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        # Decoder stage 2
        d2 = F.interpolate(
            d3,
            size=e2.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # Decoder stage 3
        d1 = F.interpolate(
            d2,
            size=e1.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        output = self.output_layer(d1)
        return output

class SpeechEnhancementCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpeechEncoder()
        self.bottleneck = SpeechBottleneck()
        self.decoder = SpeechDecoder()

    def forward(self, x):
        e1, e2, e3 = self.encoder(x)
        b = self.bottleneck(e3)
        output = self.decoder(b, e1, e2, e3)
        return output
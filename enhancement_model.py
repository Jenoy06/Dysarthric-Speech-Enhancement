{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zsLYx7KyXxVW",
        "outputId": "ae8efc25-8ee6-43c3-90bb-8970a316f953"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "PyTorch version: 2.11.0+cu130\n",
            "GPU available: Tesla T4\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "\n",
        "print(\"PyTorch version:\", torch.__version__)\n",
        "\n",
        "if torch.cuda.is_available():\n",
        "    print(\"GPU available:\", torch.cuda.get_device_name(0))\n",
        "else:\n",
        "    print(\"GPU not available. Using CPU.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "CNN ARCHITECTURE DEVELOPMENT"
      ],
      "metadata": {
        "id": "kmTxMwEJYltY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Check GPU device\n",
        "\n",
        "device = torch.device(\n",
        "    \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        ")\n",
        "\n",
        "print(\"Selected device:\", device)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NWOnbF_5YsWO",
        "outputId": "58aad84e-dc76-4edc-dd47-a7138cd11921"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Selected device: cuda\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "CNN ARCHITECTURE"
      ],
      "metadata": {
        "id": "Ojhu5Fz7ZL2R"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class ConvBlock(nn.Module):\n",
        "\n",
        "    def __init__(self, in_channels, out_channels):\n",
        "        super().__init__()\n",
        "\n",
        "        self.block = nn.Sequential(\n",
        "\n",
        "            nn.Conv2d(\n",
        "                in_channels,\n",
        "                out_channels,\n",
        "                kernel_size=3,\n",
        "                padding=1\n",
        "            ),\n",
        "\n",
        "            nn.BatchNorm2d(out_channels),\n",
        "            nn.ReLU(inplace=True),\n",
        "\n",
        "            nn.Conv2d(\n",
        "                out_channels,\n",
        "                out_channels,\n",
        "                kernel_size=3,\n",
        "                padding=1\n",
        "            ),\n",
        "\n",
        "            nn.BatchNorm2d(out_channels),\n",
        "            nn.ReLU(inplace=True)\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.block(x)"
      ],
      "metadata": {
        "id": "vzmgp7dwZXE7"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_block = ConvBlock(1, 32).to(device)\n",
        "\n",
        "test_input = torch.randn(1, 1, 80, 200).to(device)\n",
        "\n",
        "test_output = test_block(test_input)\n",
        "\n",
        "print(\"Input shape:\", test_input.shape)\n",
        "print(\"Output shape:\", test_output.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N8MTf9QVZrrH",
        "outputId": "75e1df13-0776-40bd-d44c-cc8745e5eeba"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Input shape: torch.Size([1, 1, 80, 200])\n",
            "Output shape: torch.Size([1, 32, 80, 200])\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class SpeechEncoder(nn.Module):\n",
        "\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "\n",
        "        self.enc1 = ConvBlock(1, 32)\n",
        "        self.enc2 = ConvBlock(32, 64)\n",
        "        self.enc3 = ConvBlock(64, 128)\n",
        "\n",
        "        self.pool = nn.MaxPool2d(kernel_size=2)\n",
        "\n",
        "    def forward(self, x):\n",
        "\n",
        "        e1 = self.enc1(x)\n",
        "\n",
        "        e2 = self.enc2(self.pool(e1))\n",
        "\n",
        "        e3 = self.enc3(self.pool(e2))\n",
        "\n",
        "        return e1, e2, e3"
      ],
      "metadata": {
        "id": "bBiOubg-bMvg"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "encoder = SpeechEncoder().to(device)\n",
        "\n",
        "test_input = torch.randn(1, 1, 80, 200).to(device)\n",
        "\n",
        "e1, e2, e3 = encoder(test_input)\n",
        "\n",
        "print(\"Encoder 1:\", e1.shape)\n",
        "print(\"Encoder 2:\", e2.shape)\n",
        "print(\"Encoder 3:\", e3.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TVNbvZCpbRoL",
        "outputId": "64c9d8d2-3223-47a4-95f2-12724ac7ca26"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Encoder 1: torch.Size([1, 32, 80, 200])\n",
            "Encoder 2: torch.Size([1, 64, 40, 100])\n",
            "Encoder 3: torch.Size([1, 128, 20, 50])\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class SpeechBottleneck(nn.Module):\n",
        "\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "\n",
        "        self.bottleneck = ConvBlock(128, 256)\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.bottleneck(x)"
      ],
      "metadata": {
        "id": "n_IZSmuKbovm"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "bottleneck = SpeechBottleneck().to(device)\n",
        "\n",
        "bottleneck_output = bottleneck(e3)\n",
        "\n",
        "print(\"Bottleneck output:\", bottleneck_output.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oPE7eUigbs5c",
        "outputId": "d5332fac-5815-4ff0-d954-f91a7ccf345c"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Bottleneck output: torch.Size([1, 256, 20, 50])\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class SpeechDecoder(nn.Module):\n",
        "\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "\n",
        "        self.dec3 = ConvBlock(256 + 128, 128)\n",
        "        self.dec2 = ConvBlock(128 + 64, 64)\n",
        "        self.dec1 = ConvBlock(64 + 32, 32)\n",
        "\n",
        "        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)\n",
        "\n",
        "    def forward(self, b, e1, e2, e3):\n",
        "\n",
        "        # Decoder stage 1\n",
        "        d3 = F.interpolate(\n",
        "            b,\n",
        "            size=e3.shape[2:],\n",
        "            mode=\"bilinear\",\n",
        "            align_corners=False\n",
        "        )\n",
        "\n",
        "        d3 = torch.cat([d3, e3], dim=1)\n",
        "        d3 = self.dec3(d3)\n",
        "\n",
        "        # Decoder stage 2\n",
        "        d2 = F.interpolate(\n",
        "            d3,\n",
        "            size=e2.shape[2:],\n",
        "            mode=\"bilinear\",\n",
        "            align_corners=False\n",
        "        )\n",
        "\n",
        "        d2 = torch.cat([d2, e2], dim=1)\n",
        "        d2 = self.dec2(d2)\n",
        "\n",
        "        # Decoder stage 3\n",
        "        d1 = F.interpolate(\n",
        "            d2,\n",
        "            size=e1.shape[2:],\n",
        "            mode=\"bilinear\",\n",
        "            align_corners=False\n",
        "        )\n",
        "\n",
        "        d1 = torch.cat([d1, e1], dim=1)\n",
        "        d1 = self.dec1(d1)\n",
        "\n",
        "        output = self.output_layer(d1)\n",
        "\n",
        "        return output"
      ],
      "metadata": {
        "id": "LOTpB4s1cKuK"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "decoder = SpeechDecoder().to(device)\n",
        "\n",
        "enhanced_output = decoder(\n",
        "    bottleneck_output, e1, e2, e3\n",
        ")\n",
        "\n",
        "print(\"Enhanced output shape:\", enhanced_output.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bUL5UwuMcPgE",
        "outputId": "fdc6c0fa-8d5e-4acd-e939-d12978f2a711"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Enhanced output shape: torch.Size([1, 1, 80, 200])\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class SpeechEnhancementCNN(nn.Module):\n",
        "\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "\n",
        "        self.encoder = SpeechEncoder()\n",
        "        self.bottleneck = SpeechBottleneck()\n",
        "        self.decoder = SpeechDecoder()\n",
        "\n",
        "    def forward(self, x):\n",
        "\n",
        "        # Feature extraction\n",
        "        e1, e2, e3 = self.encoder(x)\n",
        "\n",
        "        # Deep feature processing\n",
        "        b = self.bottleneck(e3)\n",
        "\n",
        "        # Spectrogram reconstruction\n",
        "        output = self.decoder(b, e1, e2, e3)\n",
        "\n",
        "        return output\n",
        "\n",
        "\n",
        "# Create complete model\n",
        "model = SpeechEnhancementCNN().to(device)\n",
        "\n",
        "print(model)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "G5HKL_FtceJm",
        "outputId": "fa516cce-9934-445c-ed90-c5f3e2c3beb5"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "SpeechEnhancementCNN(\n",
            "  (encoder): SpeechEncoder(\n",
            "    (enc1): ConvBlock(\n",
            "      (block): Sequential(\n",
            "        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): ReLU(inplace=True)\n",
            "        (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (5): ReLU(inplace=True)\n",
            "      )\n",
            "    )\n",
            "    (enc2): ConvBlock(\n",
            "      (block): Sequential(\n",
            "        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): ReLU(inplace=True)\n",
            "        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (5): ReLU(inplace=True)\n",
            "      )\n",
            "    )\n",
            "    (enc3): ConvBlock(\n",
            "      (block): Sequential(\n",
            "        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): ReLU(inplace=True)\n",
            "        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (5): ReLU(inplace=True)\n",
            "      )\n",
            "    )\n",
            "    (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
            "  )\n",
            "  (bottleneck): SpeechBottleneck(\n",
            "    (bottleneck): ConvBlock(\n",
            "      (block): Sequential(\n",
            "        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): ReLU(inplace=True)\n",
            "        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (5): ReLU(inplace=True)\n",
            "      )\n",
            "    )\n",
            "  )\n",
            "  (decoder): SpeechDecoder(\n",
            "    (dec3): ConvBlock(\n",
            "      (block): Sequential(\n",
            "        (0): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): ReLU(inplace=True)\n",
            "        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (5): ReLU(inplace=True)\n",
            "      )\n",
            "    )\n",
            "    (dec2): ConvBlock(\n",
            "      (block): Sequential(\n",
            "        (0): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): ReLU(inplace=True)\n",
            "        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (5): ReLU(inplace=True)\n",
            "      )\n",
            "    )\n",
            "    (dec1): ConvBlock(\n",
            "      (block): Sequential(\n",
            "        (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): ReLU(inplace=True)\n",
            "        (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
            "        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (5): ReLU(inplace=True)\n",
            "      )\n",
            "    )\n",
            "    (output_layer): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))\n",
            "  )\n",
            ")\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_input = torch.randn(1, 1, 80, 200).to(device)\n",
        "\n",
        "with torch.no_grad():\n",
        "    prediction = model(test_input)\n",
        "\n",
        "print(\"Input shape:\", test_input.shape)\n",
        "print(\"Prediction shape:\", prediction.shape)\n",
        "print(\"Model device:\", next(model.parameters()).device)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GLg41DuXdfj6",
        "outputId": "b9f44d31-7524-47da-c922-69966381c2bd"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Input shape: torch.Size([1, 1, 80, 200])\n",
            "Prediction shape: torch.Size([1, 1, 80, 200])\n",
            "Model device: cuda:0\n"
          ]
        }
      ]
    }
  ]
}
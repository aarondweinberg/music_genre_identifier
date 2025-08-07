import torch
import torch.nn as nn

class MelSpectrogramCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        """
        5-layer CNN for mel-spectrogram classification.
        Inputs:
            - num_classes: number of output classes (e.g., genres)
            - dropout_rate: dropout probability after each pooling
        """
        super(MelSpectrogramCNN, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Input normalization
        self.input_norm = nn.BatchNorm2d(1)

        # CNN Layers
        self.layer1 = self._make_layer(1, 64)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 128)
        self.layer4 = self._make_layer(128, 128)
        self.layer5 = self._make_layer(128, 64)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # (N, 64, 8, 4) → (N, 2048)
            nn.Linear(64 * 8 * 4, num_classes)        # Final logits
        )

    def _make_layer(self, in_channels, out_channels):
        """
        Helper to build a conv block: Conv2d + BN + ELU + MaxPool + Dropout
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        Input:  (N, 1, 256, 128)
        Output: (N, num_classes)
        """
        x = self.input_norm(x)
        x = self.layer1(x)     # → (N, 64, 128, 64)
        x = self.layer2(x)     # → (N, 128, 64, 32)
        x = self.layer3(x)     # → (N, 128, 32, 16)
        x = self.layer4(x)     # → (N, 128, 16, 8)
        x = self.layer5(x)     # → (N, 64, 8, 4)
        x = self.classifier(x) # → (N, num_classes)
        return x
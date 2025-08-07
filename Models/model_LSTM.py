import torch
import torch.nn as nn

class MelSpectrogramLSTM(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(MelSpectrogramLSTM, self).__init__()

        # Step 1: Batch normalization on the feature dimension (mel bins)
        self.batch_norm = nn.BatchNorm1d(256)  # 256 mel bins

        # Step 2: First LSTM layer: input_size = 256 mel bins, hidden_size = 128
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)

        # Step 3: Dropout after first LSTM
        self.dropout1 = nn.Dropout(dropout_rate)

        # Step 4: Second LSTM layer: hidden_size stays at 128
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Step 5: Dropout after second LSTM
        self.dropout2 = nn.Dropout(dropout_rate)

        # Step 6–8: Dense classification head
        self.fc1 = nn.Linear(128, 128)
        self.activation = nn.ELU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Input: x of shape (N, 1, 256, 128) — greyscale mel-spectrogram
        Output: logits of shape (N, num_classes)
        """
        x = x.squeeze(1)                   # (N, 1, 256, 128) → (N, 256, 128)
        x = x.permute(0, 2, 1)             # (N, 128, 256)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)             # (N, 256, 128)

        x, _ = self.lstm1(x)               # (N, 256, 128)
        x = self.dropout1(x)

        x, (hn, _) = self.lstm2(x)
        x = self.dropout2(hn.squeeze(0))   # (N, 128)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn

class GRUGenreClassifier(nn.Module):
	def __init__(self, num_classes=10, dropout_rate=0.3):
		super(GRUGenreClassifier, self).__init__()

		# Fixed model hyperparameters
		input_dim = 128		  # Frequency bins in mel spectrogram
		hidden_size = 178	  # Size of GRU hidden state
		dense_size = 256	  # Size of fully connected layers

		# LayerNorm across feature dim (128)
		self.layer_norm = nn.LayerNorm(input_dim)

		# Bidirectional GRU Layer 1 (returns full sequence)
		self.gru1 = nn.GRU(
			input_size=input_dim,
			hidden_size=hidden_size,
			batch_first=True,
			bidirectional=True
		)
		self.dropout1 = nn.Dropout(dropout_rate)

		# Bidirectional GRU Layer 2 (returns final hidden state only)
		self.gru2 = nn.GRU(
			input_size=2 * hidden_size,
			hidden_size=hidden_size,
			batch_first=True,
			bidirectional=True
		)
		self.dropout2 = nn.Dropout(dropout_rate)

		# Fully connected layers
		self.fc1 = nn.Linear(2 * hidden_size, dense_size)  # 356 → 256
		self.activation = nn.ReLU()
		self.dropout3 = nn.Dropout(dropout_rate)

		# Output layer
		self.output_layer = nn.Linear(dense_size, num_classes)	# 256 → num_classes

	def forward(self, x):
		# Input: (N, 1, 256, 128) = (batch, channel, time, frequency)
		x = x.squeeze(1)						  # (N, 256, 128) or (N, 128, 256)
		if x.shape[-1] != 128:
			x = x.transpose(1, 2)				 # Ensure (N, 256, 128)
	
		# Apply LayerNorm across feature dimension (128)	
		x = self.layer_norm(x)					 # LayerNorm over freq bins (128)

		# GRU layer 1	
		x, _ = self.gru1(x)						 # (N, 256, 356)
		x = self.dropout1(x)

		# GRU layer 2	
		_, h_n = self.gru2(x)					 # (2, N, 178)
		h_forward = h_n[0]
		h_backward = h_n[1]
		x = torch.cat((h_forward, h_backward), dim=1)  # (N, 356)
		x = self.dropout2(x)
	
		# Fully connected layers
		x = self.fc1(x)							 # (N, 256)
		x = self.activation(x)
		x = self.dropout3(x)
	
		# Output
		x = self.output_layer(x)				 # (N, num_classes)
		return x
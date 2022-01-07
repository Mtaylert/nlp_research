import torch
from torch import nn


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_dim, embedding_dim):
        super(BiLSTMRegressor, self).__init__()
        self.embeddings = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embeddings(x)
        packed_output, (hidden, cell) = self.lstm(x)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dense_outputs = self.fc(hidden)
        return dense_outputs

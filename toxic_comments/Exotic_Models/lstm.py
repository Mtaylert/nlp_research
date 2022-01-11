from torch import nn
import config
import torch
from torch.nn import functional as F

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class LSTM(nn.Module):
    def __init__(self, embedding_matrix, input_size):
        super(LSTM, self).__init__()
        embedding_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = SpatialDropout(0.3)
        self.lstm1 = nn.LSTM(embedding_size, config.LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(config.LSTM_UNITS*2, config.LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(config.DENSE_HIDDEN_UNITS, config.DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(config.DENSE_HIDDEN_UNITS, config.DENSE_HIDDEN_UNITS)

        self.linear_out = nn.Linear(config.DENSE_HIDDEN_UNITS, 1)

    def forward(self, X):
        h_embedding = self.embedding(X)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        return result


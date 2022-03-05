import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dimension = embed_size // heads

        assert (self.head_dimension * heads==embed_size), "embed size needs to be divisible by head"
        self.values = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.keys = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.queries = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #split embedding into multi-head
        values = values.reshape(N, value_len, self.heads, self.head_dimension)
        keys = keys.reshape(N, key_len, self.heads, self.head_dimension)
        query = query.reshape(N, query_len, self.heads,self.head_dimension)

        energy = torch.einsum()
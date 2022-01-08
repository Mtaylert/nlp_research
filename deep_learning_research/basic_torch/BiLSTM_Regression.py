import pandas as pd
import re
from tqdm import tqdm
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F


def build_vocab(counts):
    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)
    return vocab2index, words

def tokenize(text):
    text = str(text).lower()
    return text.split()

def build_counter(reviews):
    counts = Counter()
    for index, row in reviews.iterrows():
        counts.update(tokenize(row['Review Text']))
    return counts

def encode_sentence(text, vocab2index, N=70):
    text = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in text])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


class ReviewDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return torch.from_numpy(self.X[item][0].astype(np.int32)), torch.tensor(int(self.Y.iloc[item]), dtype=torch.float)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, embedding_dim):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.2,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)


    def forward(self, x):
        x = self.embeddings(x)

        packed_output, (hidden, cell) = self.lstm(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        dense_outputs = self.fc(hidden)
        return dense_outputs




if __name__ == '__main__':



    reviews = pd.read_csv("dataset/reviews.csv")
    counts = build_counter(reviews)
    vocab2index, words = build_vocab(counts)

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 256
    EPOCHS = 3
    VOCAB_SIZE = len(words)
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300


    reviews['encoded'] = reviews['Review Text'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))
    X = list(reviews['encoded'])
    le = LabelEncoder()
    y = reviews['Age']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    train_dataset = ReviewDataset(X_train, y_train)
    valid_dataset = ReviewDataset(X_valid, y_valid)
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    model = BiLSTM (input_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        looper = tqdm(train_dl)

        for batch_idx, (data, targets) in enumerate(looper):
            data = data.to(device)
            targets = targets.to(device)
            # forward
            age = model(data)

            loss = F.mse_loss(age, targets.unsqueeze(-1))
            print(loss)

            # backward
            optimizer.zero_grad()  # reset the gradients
            loss.backward()

            # gradient descent
            optimizer.step()

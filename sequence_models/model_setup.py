import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

def padding(tweet_int, seq_len=28):
    features = np.zeros((len(tweet_int), seq_len), dtype=int)
    for i, tweet in enumerate(tweet_int):
        if len(tweet) <= seq_len:
            zeros = list(np.zeros(seq_len - len(tweet)))
            new = tweet + zeros
        else:
            new = tweet[:seq_len]

        features[i, :] = np.array(new)
    return features

def create_data_loader(X_train,X_test,y_train,y_test,batch_size=50):
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    return train_loader,test_loader


class SentimentLSTM(nn.Module):

    def __init__(self,vocab_size, output_size, embedding_dimensions, hidden_dimensions, n_layers, drop_prob=0.5):

        super().__init__()

        self.output_size = output_size
        self.hidden_dimension = hidden_dimensions
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size,embedding_dimensions)
        self.lstm = nn.LSTM(embedding_dimensions, hidden_dimensions, n_layers,dropout=drop_prob,batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dimensions, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds,hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # Dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        sig_out = self.sigmoid(out)
        # reshape to be batch size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

    def init_hidden(self,batch_size):
        is_cuda = torch.cuda.is_available()

        # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
        if is_cuda:
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")

        h0 = torch.zeros((self.n_layers,batch_size, self.hidden_dimension)).to(device)
        c0 = torch.zeros((self.n_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

def accuracy(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred==label.squeeze()).item()


def train_loop(model, lr, clips, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    valid_loss_min = np.Inf

    epoch_tr_loss = []
    epoch_tr_acc = []

    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        model.train()

        #init hidden state
        h =  model.init_hidden(batch_size)

if __name__ =='__main__':
    # Instantiate the model w/ hyperparams
    vocab_size = 1000 + 1
    output_size = 1
    embedding_dim = 64
    hidden_dim = 256
    n_layers = 2
    model = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(model)
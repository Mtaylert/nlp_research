import sys
import numpy as np

sys.path.append("../")
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import data_setup
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BiLSTMRegressor
from torch import optim


class LSTMRegressor:
    def __init__(self, vocab_size, hidden_dim, embedding_dim, epochs):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = "saved_models/"

    def fit(self, X, y):
        train_dataset = data_setup.RegressorDatasetup(X, y)
        train_dl = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        self.model = BiLSTMRegressor(
            input_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        self.model.train()
        for epoch in range(self.epochs):
            for idx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
                data = batch["ids"]
                targets = batch["targets"]
                data = data.to(device)
                targets = targets.to(device)
                output = self.model(data)
                loss = nn.functional.mse_loss(output, targets.unsqueeze(-1))
                self.optimizer.zero_grad()  # reset the gradients
                loss.backward()
                # gradient descent
                self.optimizer.step()
            torch.save(
                self.model.state_dict(),
                "{}/{}_epoch_MODEL.model".format(self.save_path, epoch),
            )

        torch.save(
            self.model.state_dict(), "{}/LAST_MODEL.model".format(self.save_path)
        )

    def predict(self, X):
        final_outputs = []
        self.model.eval()
        test_dataset = data_setup.RegressorDatasetup(X, np.ones(len(X)))
        test_dl = DataLoader(test_dataset, batch_size=1)

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
                data = batch["ids"]
                data = data.to(device)

                outputs = self.model(data)
                final_outputs.extend(outputs.cpu().detach().numpy().tolist()[0])
        return final_outputs


if __name__ == "__main__":

    df = pd.read_csv("../dataset/reviews.csv")
    df = df[["Review Text", "Rating"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = df["Review Text"]
    y = df["Rating"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    _, vocab = config.Tokenizer(X, max_length=config.MAX_LENGTH).build_vocab()
    VOCAB_SIZE = len(vocab)

    model = LSTMRegressor(
        vocab_size=VOCAB_SIZE,
        hidden_dim=config.HIDDEN_DIM,
        embedding_dim=config.EMBEDDING_DIM,
        epochs=2,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    print(predictions, y_valid)

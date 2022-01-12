
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from data_setup import CharDataset
from model import CharacterLevelCNN


class CharacterLevelCNNModler:
    def __init__(self, epochs, optimizer="adam", lr=0.001):
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = lr
        self.optimizer = optimizer

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def fit(self, X, y):
        train_dataset = CharDataset(X,y)
        self.model = CharacterLevelCNN(input_dim=len(train_dataset.vocabulary), n_classes=1)
        self.model.to(self.device)

        train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True)

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
        self.model.train()

        for epoch in range(self.epochs):
            for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                features = batch['input'].to(self.device)
                target = batch['target'].to(self.device)
                optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.loss_fn(predictions, target)
                loss.backward()
                optimizer.step()
                scheduler.step()


    def predict(self, X):
        final_outputs = []
        self.model.eval()
        val_dataset = CharDataset(X, np.ones(len(X)))
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=True)

        with torch.no_grad():
            for idx, batch in tqdm(
                    enumerate(val_dataloader), total=len(val_dataloader)
            ):
                features = batch['input'].to(self.device)
                y_pred = self.model(features)

                final_outputs.extend(
                    torch.sigmoid(y_pred.cpu()).detach().numpy().tolist()[0]
                )
        return final_outputs

if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    df['severe_toxic'] = df.severe_toxic * 2
    df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)).astype(int)
    df['y'] = df['y'] / df['y'].max()

    df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})

    model = CharacterLevelCNNModler(epochs=2)
    model.fit(df['text'],df['y'])


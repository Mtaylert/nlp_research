import pandas as pd
from data_setup import TfidfDataset
from torch.utils.data import DataLoader
import torch
import config
from model import CNNTFIDF
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


class CNNTfidfClassifier:
    def __init__(self, epochs=2, lr=config.LEARNING_RATE):

        self.epochs = epochs
        self.tfidf = config.TFIDF
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

    def loss_fn(self, outputs, targets):

        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def fit(self, X, y):
        self.tfidf_modeler = self.tfidf.fit(X)
        input_size = len(tfidf_modeler.get_feature_names())
        train_dataset = TfidfDataset(
            input_text=X, targets=y, tfidf_modeler=self.tfidf_modeler
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4
        )
        self.model = CNNTFIDF(
            input_size=input_size,
            hidden1=config.HIDDEN1,
            hidden2=config.HIDDEN2,
            num_labels=1,
        )
        self.model.to(self.device)
        self.model.train()
        self.parameters = self.model.parameters()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters),
            lr=self.lr,
        )
        scheduler = CosineAnnealingLR(optimizer, 1)

        for epoch in range(self.epochs):

            for batch_idx, batch in tqdm(
                enumerate(train_dataloader), total=len(train_dataloader)
            ):

                features = batch["ids"]
                # add a new dimension: channel to enter cnn
                features = features.unsqueeze(dim=2)

                targets = batch["targets"]

                features = features.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.float)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

    def predict(self, X):

        final_outputs = []
        test_dataset = TfidfDataset(
            input_text=X, targets=np.ones(len(X)), tfidf_modeler=self.tfidf_modeler
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4
        )

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(test_dataloader), total=len(test_dataloader)
            ):
                features = batch["ids"]
                # add a new dimension: channel to enter cnn
                features = features.unsqueeze(dim=2)
                features = features.to(self.device, dtype=torch.float)
                predictions = self.model(features)
                final_outputs.extend(
                    torch.sigmoid(predictions.cpu()).detach().numpy().tolist()[0]
                )
        return final_outputs


if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/matthew/Documents/nlp_training/nlp_research/toxic_comments/data/train.csv"
    )
    df = df.sample(frac=0.05)
    df = df.reset_index(drop=True)
    X = df["comment_text"]
    y = df["severe_toxic"]
    clf = CNNTfidfClassifier(epochs=2, lr=config.LEARNING_RATE)
    clf.fit(X, y)

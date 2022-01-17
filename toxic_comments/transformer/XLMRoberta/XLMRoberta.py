from data_setup import XLMRobertaDataset
from model import XLMRoberta
from torch import nn
import torch
from torch.utils.data import DataLoader
from data_setup import XLMRobertaDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import numpy as np


class XLMClassifier:
    def __init__(self, epochs):
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loss_fn(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def fit(self, X, y):
        train_dataset = XLMRobertaDataset(X, y)
        train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)

        self.model = XLMRoberta()
        self.model.to(self.device)
        self.model.train()

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        num_train_steps = int(len(X) / 32 * 3)
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )

        for epoch in range(self.epochs):

            for batch_idx, dataset in tqdm(
                    enumerate(train_dataloader), total=len(train_dataloader)
            ):
                ids = dataset["ids"]
                mask = dataset["mask"]
                token_type_ids = dataset["token_type_ids"]
                targets = dataset["targets"]


                ids = ids.to(self.device, dtype=torch.long)
                mask = mask.to(self.device, dtype=torch.long)
                token_type_ids = token_type_ids.to(self.device, dtype=torch.long)
                targets = targets.to(self.device, dtype=torch.float)

                optimizer.zero_grad()
                predictions = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids)

                loss = self.loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

    def predict(self, X):
        final_outputs = []

        val_dataset = XLMRobertaDataset(X, np.ones(len(X)))
        val_dataloader = DataLoader(val_dataset)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, dataset in tqdm(
                    enumerate(val_dataloader), total=len(val_dataloader)
            ):
                ids = dataset["ids"]
                mask = dataset["mask"]
                token_type_ids = dataset["token_type_ids"]

                ids = ids.to(self.device, dtype=torch.long)
                mask = mask.to(self.device, dtype=torch.long)
                token_type_ids = token_type_ids.to(self.device, dtype=torch.long)

                outputs = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids)
                final_outputs.extend(
                    torch.sigmoid(outputs.cpu()).detach().numpy().tolist()[0]
                )
        return final_outputs




if __name__ == '__main__':
    df = pd.read_csv(
        "/Users/matthew/Documents/nlp_training/nlp_research/toxic_comments/data/train.csv"
    )
    df = df.sample(frac=0.05)
    df = df.reset_index(drop=True)
    X = df["comment_text"]
    y = df["severe_toxic"]
    clf = XLMClassifier(epochs=2)
    clf.fit(X,y)




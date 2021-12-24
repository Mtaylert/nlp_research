import config
from model import BertBaseUncased
from load_data import load_data
import pandas as pd
import numpy as np
import data_setup

from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
from tqdm import tqdm


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train():
    save_path = "model/"
    df = load_data()
    train = df[df["partition"] == "train"]
    dev = df[df["partition"] == "dev"]
    test = df[df["partition"] == "test"]

    train_dataset = data_setup.BertDatasetTraining(
        input1=train["sentence_1"],
        input2=train["sentence_2"],
        target=train["similarity"],
    )

    dev_dataset = data_setup.BertDatasetTraining(
        input1=dev["sentence_1"], input2=dev["sentence_2"], target=dev["similarity"]
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )
    valid_data_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertBaseUncased()
    model.to(device)
    param_optimizer = list(model.named_parameters())
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

    num_train_steps = int(len(train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    model.train()
    for epoch in range(config.EPOCHS):
        for batch_idx, dataset in tqdm(
            enumerate(train_data_loader), total=len(train_data_loader)
        ):
            ids = dataset["ids"]
            mask = dataset["mask"]
            token_type_ids = dataset["token_type_ids"]
            targets = dataset["targets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(ids, mask=mask, token_type_ids=token_type_ids)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

    torch.save(model.state_dict(), "{}/LAST_MODEL.model".format(save_path))
    return model


if __name__ == "__main__":
    model = train()

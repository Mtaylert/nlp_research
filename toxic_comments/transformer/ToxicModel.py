import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import transformers
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
from tqdm import tqdm


params = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'debug': False,
    'model': '/kaggle/input/roberta-base-model',
    'output_logits': 768,
    'max_len': 256,
    'batch_size': 32,
    'dropout': 0.2,
    'num_workers': 2,
    'n_folds':5,
    'frac1':0.3,
    'frac1_factor':1.2,
    'save_path':'/kaggle/working/'
}


class ToxicDataset:
    def __init__(self, input1,  target):
        self.input1 = input1
        self.target = target
        self.TOKENIZER = AutoTokenizer.from_pretrained(params['model'],do_lower_case=True)

    def __len__(self):
        return len(self.input1)

    def __getitem__(self, item):
        input1 = str(self.input1[item])

        input1 = " ".join(input1.split())

        inputs = self.TOKENIZER.encode_plus(
            input1,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=params['max_len'],
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(int(self.target[item]), dtype=torch.long),
        }



class ToxicityModel(nn.Module):
    def __init__(self, checkpoint=params['model'], params=params):
        super(ToxicityModel, self).__init__()
        self.checkpoint = checkpoint
        self.bert = AutoModel.from_pretrained(checkpoint, return_dict=False)
        self.layer_norm = nn.LayerNorm(params['output_logits'])
        self.dropout = nn.Dropout(params['dropout'])
        self.dense = nn.Sequential(
            nn.Linear(params['output_logits'], 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(params['dropout']),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        preds = self.dense(pooled_output)
        return preds


class ToxicComments:
    """
    Passing the transformer architecture into a sklearn arch
    """

    def __init__(self, epochs: int = 3, retrain: bool = True, num=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.save_path = "/kaggle/output/"
        self.model = ToxicityModel()
        if retrain:
            self.model.to(self.device)
        else:

            state = torch.load(params['save_path'] + f"{num}_fold.model", map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)

    def loss_fn(self, outputs, targets):

        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def fit(self, X, y, num=0):
        train_dataset = ToxicDataset(
            X,
            y
        )

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=params['batch_size'], num_workers=4
        )

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

        num_train_steps = int(len(X) / params['batch_size'] * 3)
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )
        self.model.train()

        for epoch in range(self.epochs):
            for batch_idx, dataset in tqdm(
                    enumerate(train_data_loader), total=len(train_data_loader)
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
                outputs = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids)

                loss = self.loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

        torch.save(
            self.model.state_dict(), "{}/{}_fold.model".format(params['save_path'], num)
        )

    def predict(self, X):
        final_outputs = []
        self.model.eval()

        val_dataset = ToxicDataset(
            X,
            np.ones(len(X))
        )

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=4
        )

        with torch.no_grad():
            for batch_idx, dataset in tqdm(
                    enumerate(val_data_loader), total=len(val_data_loader)
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

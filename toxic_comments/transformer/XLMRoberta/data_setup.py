import pandas as pd
import transformers
import torch
from torch.utils.data import DataLoader

class XLMRobertaDataset:

    def __init__(self, inputs, targets):
        self.input = inputs
        self.targets = targets
        self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        input_item = str(self.input[item])

        input_item = " ".join(input_item.split())
        tokenized_inputs = self.tokenizer.encode_plus(
            input_item,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        ids = tokenized_inputs["input_ids"]
        token_type_ids = tokenized_inputs["token_type_ids"]
        mask = tokenized_inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(int(self.targets[item]), dtype=torch.long),
        }




if __name__ == '__main__':
    df = pd.read_csv(
            "/Users/matthew/Documents/nlp_training/nlp_research/toxic_comments/data/train.csv"
        )
    df = df.sample(frac=0.05)
    df = df.reset_index(drop=True)
    X = df["comment_text"]
    y = df["severe_toxic"]
    traindataset = XLMRobertaDataset(X, y)
    traindataloader = DataLoader(traindataset, batch_size=32)

    for bb in traindataloader:
        print(bb)
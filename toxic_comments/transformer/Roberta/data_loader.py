import pandas as pd
import torch
import transformers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = {
    'device': device,
    'debug': False,
    'model': 'roberta-base',
    'output_logits': 768,
    'max_len': 256,
    'batch_size': 32,
    'dropout': 0.2,
    'num_workers': 2
}

class ToxicDataset:
    def __init__(self, input1,  target):
        self.input1 = input1
        self.target = target
        self.TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

    def __len__(self):
        return len(self.input1)

    def __getitem__(self, item):
        input1 = str(self.input1[item])

        input1 = " ".join(input1.split())

        inputs = self.TOKENIZER.encode_plus(
            input1,
            add_special_tokens=True,
            max_length=params['max_len'],
            padding="max_length",
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

if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    train_dataset = ToxicDataset(df['comment_text'], df['toxic'])
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params['batch_size'], num_workers=4
    )

    for _ in train_data_loader:
        print(_)

import pandas as pd
import transformers
import torch
import config
from torch.utils.data import DataLoader


def create_text_feature(df):
    text_combined = []
    for (i,row) in df.iterrows():
        combined = ''
        for txt_col in ['category','text','summary']:
            combined +=(str(row[txt_col])+ '[SEP] ')
        text_combined.append(combined)
    df['concat_text'] = text_combined
    return df


class AmazonDataset():
    def __init__(self,text, target):
        self.text = text
        self.target = target
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.MODEL, do_lower_case=True)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        input = str(self.text[item])

        inputs = self.tokenizer(
            input,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            truncation=True,
            padding='max_length',
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
    df = pd.read_csv('amazon_de_reviews_small.csv')
    df = create_text_feature(df)
    german_dl = torch.utils.data.DataLoader(AmazonDataset(df['text'],df['rating']), batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)
    for idx, _ in enumerate(german_dl):
        print(_)

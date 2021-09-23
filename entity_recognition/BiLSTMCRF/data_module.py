from typing import List
import config
import data_reader
import torch
from torch.utils.data import DataLoader


class BLSTMCRFDataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text_row = ' '.join(self.text[idx])
        target_row  = self.target[idx]

        encoding_inputs = self.tokenizer(
            text_row,
            max_length=self.max_len,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        ids = encoding_inputs['input_ids']
        mask = encoding_inputs['attention_mask']
        token_type_ids = encoding_inputs['token_type_ids']
        target_row = target_row + ([0] * (config.MAX_LEN-len(target_row)))

        return {
            "ids": ids,
            "mask": mask,
            "token_type_ids": token_type_ids,
            "targets": torch.tensor(target_row, dtype=torch.float)
        }

if __name__ == '__main__':
    train_sent, train_tags = data_reader.process_csv(config.TRAIN_DATA)
    train_dataset = BLSTMCRFDataset(train_sent,train_tags)



    for batch in train_dataset:
        print(batch)
        break
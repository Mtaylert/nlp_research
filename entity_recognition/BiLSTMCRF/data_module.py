from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

import config
import data_reader


class BLSTMCRFDataset:
    def __init__(self, text: List[str], target: List[str]):
        self.text = text
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.text[idx]
        tags = self.target[idx]

        ids = []
        target_tag = []

        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(s, add_special_tokens=False)
            # abhishek: ab ##hi ##sh ##ek
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[: config.MAX_LEN - 2]
        target_tag = target_tag[: config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(target_tag, dtype=torch.long),
        }


if __name__ == "__main__":
    train_sent, train_tags = data_reader.process_csv(config.TRAIN_DATA)
    train_dataset = BLSTMCRFDataset(train_sent, train_tags)

    for batch in train_dataset:
        print(batch)
        break

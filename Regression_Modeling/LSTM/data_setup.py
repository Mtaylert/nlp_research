import torch
import config


class RegressorDatasetup:
    def __init__(self, text, targets):
        self.text = text
        self.targets = targets
        self.tokenizer = config.Tokenizer(self.text)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        self.target = list(self.targets)[item]
        self.input = list(self.text)[item]
        encoding = self.tokenizer.encode(self.input)
        return {
            "ids": torch.tensor(encoding, dtype=torch.long),
            "targets": torch.tensor(self.target, dtype=torch.float),
        }

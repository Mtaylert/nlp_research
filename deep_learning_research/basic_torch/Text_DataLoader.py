import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import  pad_sequence
from torch.utils.data import Dataset, DataLoader


class ReviewsDataset(Dataset):
    def __init__(self, text):
        self.text = text

        self.vocab = Vocabulary()
        self.vocab.build_vocabulary(self.text)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text_record = self.text[item]

        tokenized = [self.vocab.stoi["<SOS>"]]
        tokenized += self.vocab.convert(text_record)
        tokenized.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(tokenied)
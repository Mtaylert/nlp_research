import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

class CharDataset:
    def __init__(self, input_text, target, max_length=1014):
        self.input_text = input_text
        self.target = target
        self.vocabulary = self.build_vocab()
        self.identity_mat = np.identity(len(self.vocabulary))

        char_text_list = []
        for doc in tqdm(self.input_text):
            sub_text = ""
            for text in doc.split():
                for char in text:
                    sub_text += char
                    sub_text += " "
            char_text_list.append(sub_text)

        self.char_text = char_text_list
        self.max_length = max_length
        self.n_classes = len(set(self.target))

    def build_vocab(self):
        char_vocab = {}
        for doc in tqdm(self.input_text):
            for token in doc.split():
                for char in token:
                    if char in char_vocab:
                        char_vocab[char] += 1
                    else:
                        char_vocab[char] = 1
        return list(char_vocab.keys())

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        raw_text = self.char_text[index]
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
            label = self.target[index]
        return {'input':data, 'target':label}






if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')

    training_set = CharDataset(df['comment_text'],df['severe_toxic'])
    training_params = {"batch_size": 32,
                       "shuffle": True,
                       "num_workers": 0}
    training_generator = DataLoader(training_set, **training_params)


    for bb in training_generator:
        print(bb)

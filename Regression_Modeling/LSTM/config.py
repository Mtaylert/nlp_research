from collections import Counter
import numpy as np
import pandas as pd

BATCH_SIZE = 256
EPOCHS = 3
LEARNING_RATE = 0.001
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
MAX_LENGTH = 70
N_FOLDS = 2
FRAC1 = 0.3
FRAC1_FACTOR = 1.2
FOLD_PATH = 'folds/'

class Tokenizer:
    def __init__(self, text, max_length: int = 70):
        self.text = text
        self.max_length = max_length
        self.counts = self.build_counter()
        self.vocab2index, self.words = self.build_vocab()

    def build_counter(self):
        counts = Counter()
        for index, line in enumerate(self.text):
            counts.update(self.tokenize(line))
        return counts

    def tokenize(self, text):
        tokenized_text = str(text).lower().split()
        return tokenized_text

    def build_vocab(self):
        vocab2index = {"": 0, "UNK": 1}
        words = ["", "UNK"]
        for word in self.counts:
            vocab2index[word] = len(words)
            words.append(word)
        return vocab2index, words

    def encode(self, text):
        encoded = np.zeros(self.max_length, dtype=int)
        enc1 = np.array(
            [
                self.vocab2index.get(word, self.vocab2index["UNK"])
                for word in self.tokenize(text)
            ]
        )
        length = min(self.max_length, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded



class BuildFolds:

    def __init__(self, df, n_folds, frac1, frac1_factor, save_path):
        self.df = df
        self.n_folds = n_folds
        self.frac1 = frac1
        self.frac1_factor = frac1_factor
        self.save_path = save_path

    def run(self):
        for fld in range(self.n_folds):
            tmp_df = pd.concat([self.df[self.df.y > 0].sample(frac=self.frac1, random_state=10 * (fld + 1)),
                                self.df[self.df.y == 0].sample(
                                    n=int(len(self.df[self.df.y > 0]) * self.frac1 * self.frac1_factor),
                                    random_state=10 * (fld + 1))], axis=0).sample(frac=1, random_state=10 * (fld + 1))

            tmp_df.to_csv(self.save_path + f'df_fld{fld}.csv', index=False)

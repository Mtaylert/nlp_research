from collections import Counter
import numpy as np

BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.001
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
MAX_LENGTH = 70


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

import numpy as np
from keras.preprocessing import text, sequence
from tqdm import tqdm
import gc
import config
import pandas as pd

class Embedder:

    def __init__(self, input_text):
        self.input_text = input_text
        self.tokenizer = text.Tokenizer()
        self.tokenizer.fit_on_texts(list(self.input_text))

        self.crawl_embeddings, self.unknown_words_crawl = self.build_matrix(config.CRAWL_EMBEDDINGS, self.tokenizer.word_index)
        self.glove_embeddings, self.unknown_words_glove = self.build_matrix(config.GLOVE_EMBEDDINGS, self.tokenizer.word_index)

        self.max_features =  len(self.tokenizer.word_index) + 1

        self.embedding_matrix = np.concatenate([self.crawl_embeddings, self.glove_embeddings], axis=-1)
        self.unknowns = {'crawl':self.unknown_words_crawl, 'glove':self.unknown_words_glove}
        self.cache = {'embedding_matrix':self.embedding_matrix, 'max_features':self.max_features}

    def get_coefficients(self, word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def load_embeddings(self, path):
        with open(path) as f:
            return dict(self.get_coefficients(*line.strip().split(' ')) for line in tqdm(f))

    def build_matrix(self, path, word_index):
        embedding_index = self.load_embeddings(path)
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        unknown_words = []

        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                unknown_words.append(word)
        return embedding_matrix, unknown_words



if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    df['severe_toxic'] = df.severe_toxic * 2
    df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)).astype(int)
    df['y'] = df['y'] / df['y'].max()
    df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
    embedding_dict = Embedder(df['text']).cache


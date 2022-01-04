from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


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


class TextEmbedder:
    def __init__(self, dimensions=50, window=5, sg=1, epochs=3, min_n=2, max_n=6):
        self.fasttext_params = {'vector_size':dimensions,
                                'window':window,
                                'sg':sg,
                                'min_count':1,
                                'epochs':epochs,
                                'min_n':min_n,
                                'max_n':max_n}

    def fit(self, text):
        self.fast_text = FastText(text, **self.fasttext_params)

    def transform(self, text):
        fast_embeddings = {}
        for idx, sent in enumerate(text):
            sub_embedding = []
            for word in enumerate(sent.split()):
                sub_embedding.append(self.fast_text.wv[word])
            fast_embeddings[idx] = np.sum(sub_embedding, axis=0)[0]

        fast_df = pd.DataFrame.from_dict(fast_embeddings,orient='index')

        return fast_df

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    embedding_module = TextEmbedder()
    embedding_module.fit(df['comment_text'])
    embedding_df = embedding_module.transform(df['comment_text'])
    print(embedding_df)

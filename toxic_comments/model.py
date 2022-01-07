from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')


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


def clean(data, col):
    data[col] = data[col].str.replace(r"what's", "what is ")
    data[col] = data[col].str.replace(r"\'ve", " have ")
    data[col] = data[col].str.replace(r"can't", "cannot ")
    data[col] = data[col].str.replace(r"n't", " not ")
    data[col] = data[col].str.replace(r"i'm", "i am ")
    data[col] = data[col].str.replace(r"\'re", " are ")
    data[col] = data[col].str.replace(r"\'d", " would ")
    data[col] = data[col].str.replace(r"\'ll", " will ")
    data[col] = data[col].str.replace(r"\'scuse", " excuse ")
    data[col] = data[col].str.replace(r"\'s", " ")

    # Clean some punctutations
    data[col] = data[col].str.replace('\n', ' \n ')
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)', r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}', r'\1\1\1')
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)', r' \1 ')
    # patterns with repeating characters
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b', r'\1\1')
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B', r'\1\1\1')
    data[col] = data[col].str.replace(r'[ ]{2,}', ' ').str.strip()
    data[col] = data[col].str.replace(r'[ ]{2,}', ' ').str.strip()
    data[col] = data[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return data

class TextEmbedder:
    def __init__(self, dimensions=300, window=7, sg=1, epochs=10, min_n=1, max_n=6):
        self.fasttext_params = {'vector_size':dimensions,
                                'window':window,
                                'sg':sg,
                                'min_count':1,
                                'epochs':epochs,
                                'min_n':min_n,
                                'max_n':max_n}

    def fit(self, text):
        self.fast_text = FastText(text, **self.fasttext_params)
        self.fast_text.save("FastText.model")

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
    df =clean(df, 'comment_text')

    mod = TextEmbedder()
    mod.fit(df['comment_text'])
    #fmodel = FastText.load('FastText.model')
    #vecs = fmodel.wv[df['comment_text'].iloc[0].split()].mean(axis=0)
    #print(vecs)


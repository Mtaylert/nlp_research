import pandas as pd
import numpy as np
import torch
from keras import Model, Input
from keras.layers import LSTM, Dense, Embedding, Flatten, SpatialDropout1D, TimeDistributed, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        agg_func = lambda s: [(word,pos,tag) for word, pos,tag in zip(s['Word'],
                                                                      s['POS'],
                                                                      s['Tag'])]

        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]


def id_word_tag(words,tags):
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    return word2idx,tag2idx


class NER_Setup():
    def  __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def preprocess(self):
        self.data = self.data.fillna(method='ffill')
        words = list(set(self.data['Word']))
        words.append("ENPAD")
        self.num_words = len(words)
        tags = list(set(self.data['Tag'].values))
        self.num_tags = len(tags)
        word2idx, tag2idx = id_word_tag(words, tags)
        getter = SentenceGetter(self.data)
        self.sentences = getter.sentences

        self.X = [[word2idx[w[0]] for w in s] for s in self.sentences]
        self.X = pad_sequences(maxlen=self.max_len, sequences=self.X, padding='post', value=self.num_words + 1)

        self.y = [[tag2idx[w[2]] for w in s] for s in self.sentences]
        self.y = pad_sequences(maxlen=self.max_len, sequences=self.y, padding='post', value=tag2idx['O'])
        self.y = [to_categorical(i, num_classes=self.num_tags) for i in self.y]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=123, test_size=.2)

    def build_model(self):

        input_word = Input(shape=(self.max_len,))
        model = Embedding(input_dim=self.num_words, output_dim=self.max_len, input_length=self.max_len)(input_word)
        model = SpatialDropout1D(0.1)(model)
        model = Bidirectional(LSTM(units=256, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(self.num_tags, activation="softmax"))(model)
        model = Model(input_word, out)
        print(model.summary())
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])





if __name__ == '__main__':
    data = pd.read_csv('ner_dataset.csv', encoding='latin1')
    MAX_LEN = 50
    NER = NER_Setup(data = data, max_len=MAX_LEN)
    NER.preprocess()
    NER.build_model()
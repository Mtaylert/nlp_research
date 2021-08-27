import re

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Embedding, Flatten, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical


class LstmModel:
    def __init__(self, text, max_features, embed_dim, lstm_out):
        self.text = text
        self.max_features = max_features
        self.max_len = 30
        self.embed_dim = embed_dim
        self.lstm_out = lstm_out
        self.tokenizer = Tokenizer(num_words=self.max_features, split=" ")

    def prepare(self):
        self.tokenizer.fit_on_texts(self.text)
        self.X = self.tokenizer.texts_to_sequences(self.text)
        self.X = pad_sequences(self.X, maxlen=self.max_len)
        return self.X

    def build_model(self):
        model = Sequential()
        model.add(
            Embedding(self.max_features, self.embed_dim, input_length=self.X.shape[1])
        )
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        print(model.summary())
        return model

    def inference(self, text, model):
        seq = self.tokenizer.texts_to_sequences(text)
        seq = pad_sequences(seq, maxlen=self.max_len, dtype='int32',value=0)
        sentiment = model.predict(seq, batch_size=1, verbose=2)[0]
        if np.argmax(sentiment) == 0:
            print("negative",sentiment)
        elif np.argmax(sentiment) == 1:
            print("positive",sentiment)

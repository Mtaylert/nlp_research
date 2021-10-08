import gensim
from gensim.models import Word2Vec, FastText
import pandas as pd
from sklearn import preprocessing, model_selection, metrics
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List

class EmbedText:
    def __init__(self, text:List, embedding_modeler: gensim.models, size: int =150):
        self.text= text
        self.size = size
        self.model = embedding_modeler(sentences=self.text, sg=1, vector_size=self.size, min_count=0, epochs=20)
    def build_embeding_dict(self) -> dict:
        embedding_dict = {}
        for idx, text_line in enumerate(self.text):
            embedding_list = [self.model.wv[word] for word in text_line]
            embedding_dict[idx] = np.sum(embedding_list, axis=0)
        return embedding_dict

if __name__ == '__main__':
    data = pd.read_csv('../data.csv')
    data.Title = data.Title.str.lower()
    data.Title = data.Title.apply(lambda x: x.split())
    enc_labeler = preprocessing.LabelEncoder()
    Y = enc_labeler.fit_transform(data.Conference)
    text = data.Title

    X = EmbedText(text, embedding_modeler=FastText).build_embeding_dict()
    X_train,X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=.15,random_state=123)
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train,y_train)
    print(metrics.accuracy_score(y_test,clf.predict(X_test)))
import pandas as pd
import string
from collections import Counter
import numpy as np
from nltk.corpus import stopwords, twitter_samples
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from model_setup import LstmModel

stop_words = set(stopwords.words('english'))


def load_tweets():
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    return all_positive_tweets, all_negative_tweets

def data_preprocessing(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text) # Remove HTML from text
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    return text






if __name__ =='__main__':
    all_positive_tweets, all_negative_tweets = load_tweets()
    data_pos = pd.DataFrame({"target": np.ones(len(all_positive_tweets)), "text": all_positive_tweets})
    data_neg = pd.DataFrame({"target": np.zeros(len(all_negative_tweets)), "text": all_negative_tweets})
    data = shuffle(pd.concat([data_pos, data_neg]))
    data['clean_tweets'] = data['text'].apply(data_preprocessing)
    sequence_model = LstmModel(text=data['clean_tweets'], max_features=2000, embed_dim=256, lstm_out=324)
    features = sequence_model.prepare()
    model = sequence_model.build_model()
    Y = pd.get_dummies(data['target']).values
    X_train,X_test, y_train, y_test = train_test_split(features, Y, test_size=.2, random_state=42)
    batch_size = 32
    model.fit(X_train, y_train, epochs= 10, batch_size=batch_size, verbose=2)
    score, acc = model.evaluate(X_test,y_test, verbose=2, batch_size=batch_size)
    print(score,acc)
    print('--------------------------------------------------------------')

    tweet = ['United Sates foreign policy is a disaster']
    sequence_model.inference(tweet, model)
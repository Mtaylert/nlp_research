import pandas as pd
import string
from collections import Counter
import numpy as np
from nltk.corpus import stopwords, twitter_samples
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from model_setup import padding,create_data_loader


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


def get_text_int(data):
    corpus = [word for text in data['clean_tweets'] for word in text.split()]
    count_words = Counter(corpus)
    sorted_words = count_words.most_common()
    vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}

    tweet_int = []
    for text in data['clean_tweets']:
        r = [vocab_to_int[word] for word in text.split()]
        tweet_int.append(r)

    data['tweet_int'] = tweet_int
    tweet_len = [len(x) for x in tweet_int]
    data['tweet_len'] = tweet_len
    return data,tweet_int




if __name__ =='__main__':
    all_positive_tweets, all_negative_tweets = load_tweets()
    data_pos = pd.DataFrame({"target": np.ones(len(all_positive_tweets)), "text": all_positive_tweets})
    data_neg = pd.DataFrame({"target": np.zeros(len(all_negative_tweets)), "text": all_negative_tweets})
    data = shuffle(pd.concat([data_pos, data_neg]))
    data['clean_tweets'] = data['text'].apply(data_preprocessing)
    data,tweet_int = get_text_int(data)
    features = padding(tweet_int, 28)
    X_train,X_test,y_train,y_test = train_test_split(features,data['target'],test_size=.2,random_state=123)
    print(X_train)
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords


def read_dataset(filepath):
    enc_tag = preprocessing.LabelEncoder()

    df = pd.read_csv(filepath)
    df.loc[:, "Conference"] = enc_tag.fit_transform(df["Conference"])
    X_train, X_val, y_train, y_val = train_test_split(df.Title.values,
                                                      df.Conference.values,
                                                      test_size=0.15,
                                                      random_state=42,
                                                      stratify=df.Conference.values)

    return X_train, X_val, y_train, y_val,enc_tag

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text = [word for word in text.split() if word not in stop_words and len(word)>=2]
    return ' '.join(text)


if __name__ == '__main__':

    X_train, X_val, y_train, y_val, enc_tag = read_dataset('data.csv')
    print(X_train)



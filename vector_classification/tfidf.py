import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def text_preprocess(text: str) -> str:
    """
    Apply NLP Preprocessing Techniques to the reviews.
    """
    main_words = re.sub("[^a-zA-Z]", " ", text)  # Retain only alphabets
    main_words = (main_words.lower()).split()
    main_words = [
        w for w in main_words if not w in set(stopwords.words("english"))
    ]  # Remove stopwords

    lem = WordNetLemmatizer()
    main_words = [
        lem.lemmatize(w) for w in main_words if len(w) > 1
    ]  # Group different forms of the same word

    main_words = " ".join(main_words)
    return main_words


def format_text_for_model(data: pd.DataFrame):
    Y = data["Recommended IND"]
    X = [text_preprocess(x) for x in data["Review Text"]]
    tfidf = TfidfVectorizer()
    X_features = tfidf.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, Y, random_state=123, shuffle=True, stratify=Y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    reviews = pd.read_csv(
        "https://raw.githubusercontent.com/hanzhang0420/Women-Clothing-E-commerce/master/Womens%20Clothing%20E-Commerce%20Reviews.csv"
    )
    reviews = reviews[["Review Text", "Recommended IND"]].dropna()
    X_train, X_test, y_train, y_test = format_text_for_model(reviews)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print(confusion_matrix(clf.predict(X_test), y_test))

import spacy
import pandas as pd
from reviews import SpacySetup


def infer_result(text):
    nlp = spacy.load("output/model-best")
    doc = nlp(text[0])
    return doc


reviews = pd.read_csv(
    "https://raw.githubusercontent.com/hanzhang0420/Women-Clothing-E-commerce/master/Womens%20Clothing%20E-Commerce%20Reviews.csv"
)
reviews = reviews[["Review Text", "Recommended IND"]].dropna()

model = SpacySetup(reviews)
for r in range(20000,21000):
    result = infer_result(model.zipped_data[r])
    print(result.cats)
    print(model.zipped_data[r])
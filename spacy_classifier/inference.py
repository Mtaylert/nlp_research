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
result = infer_result(model.zipped_data[21002])
print(result.cats)
print(model.zipped_data[21002])
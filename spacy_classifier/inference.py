import spacy
import pandas as pd
from reviews import SpacySetup


def infer_result(text):
    nlp = spacy.load("output/model-best")
    doc = nlp(text[0])
    return doc




model = SpacySetup('this is a review')
print(model)

import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm


class SpacySetup:
    def __init__(self, data: pd.DataFrame, spacy_model: str = None, build_docs=False):
        if spacy_model:
            self.nlp = spacy.load(spacy_model)
        else:
            self.nlp = spacy.load("en_core_web_sm")

        self.data = data
        self.zipped_data = self.zip_dataset()
        if build_docs:
            self.spacy_docs = self.make_spacy_docs()

    def __repr__(self):
        return self.data

    def zip_dataset(self):
        zipped_data = tuple(
            zip(
                self.data["Review Text"].tolist(), self.data["Recommended IND"].tolist()
            )
        )
        return zipped_data

    def make_spacy_docs(self):
        docs = []

        for doc, label in tqdm(self.nlp.pipe(self.zipped_data, as_tuples=True)):
            if label == 1:
                doc.cats["positive"] = 1
                doc.cats["negative"] = 0

            else:
                doc.cats["positive"] = 0
                doc.cats["negative"] = 1

            docs.append(doc)
        return docs


if __name__ == "__main__":
    reviews = pd.read_csv("https://raw.githubusercontent.com/hanzhang0420/Women-Clothing-E-commerce/master/Womens%20Clothing%20E-Commerce%20Reviews.csv")
    reviews = reviews[["Review Text", "Recommended IND"]].dropna()

    model = SpacySetup(reviews, build_docs=True)

    doc_bin = DocBin(docs=model.spacy_docs[:15000])
    doc_bin.to_disk("data/train.spacy")

    doc_bin = DocBin(docs=model.spacy_docs[15000:20000])
    doc_bin.to_disk("data/valid.spacy")

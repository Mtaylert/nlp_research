import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.cli import download
download("en_core_web_trf")

class SpacySetup:
    def __init__(self, data: pd.DataFrame, spacy_model: str = None, build_docs=False):
        if spacy_model:
            self.nlp = spacy.load(spacy_model)
        else:
            self.nlp = spacy.load("en_core_web_trf")

        self.data = data
        self.zipped_data = self.zip_dataset()
        if build_docs:
            self.spacy_docs = self.make_spacy_docs()

    def __repr__(self):
        return self.data

    def zip_dataset(self):
        zipped_data = tuple(
            zip(
                self.data["Title"].tolist(), self.data["Conference"].tolist()
            )
        )
        return zipped_data

    def make_spacy_docs(self):
        docs = []

        for doc, label in tqdm(self.nlp.pipe(self.zipped_data, as_tuples=True)):
            if label == 'ISCAS':
                doc.cats["ISCAS"] = 1
                doc.cats["INFOCOM"] = 0
                doc.cats["VLDB"] = 0
                doc.cats["WWW"] = 0
                doc.cats["SIGGRAPH"] = 0

            elif label == 'INFOCOM':
                doc.cats["ISCAS"] = 0
                doc.cats["INFOCOM"] = 1
                doc.cats["VLDB"] = 0
                doc.cats["WWW"] = 0
                doc.cats["SIGGRAPH"] = 0

            elif label == 'VLDB':
                doc.cats["ISCAS"] = 0
                doc.cats["INFOCOM"] = 0
                doc.cats["VLDB"] = 1
                doc.cats["WWW"] = 0
                doc.cats["SIGGRAPH"] = 0

            elif label == 'WWW':
                doc.cats["ISCAS"] = 0
                doc.cats["INFOCOM"] = 0
                doc.cats["VLDB"] = 0
                doc.cats["WWW"] = 1
                doc.cats["SIGGRAPH"] = 0

            elif label == 'SIGGRAPH':
                doc.cats["ISCAS"] = 0
                doc.cats["INFOCOM"] = 0
                doc.cats["VLDB"] = 0
                doc.cats["WWW"] = 0
                doc.cats["SIGGRAPH"] = 1

            docs.append(doc)
        return docs


if __name__ == "__main__":
    reviews = pd.read_csv("data.csv")
    model = SpacySetup(reviews, build_docs=True)

    doc_bin = DocBin(docs=model.spacy_docs[:1500])
    doc_bin.to_disk("data/train.spacy")

    doc_bin = DocBin(docs=model.spacy_docs[1500:])
    doc_bin.to_disk("data/valid.spacy")

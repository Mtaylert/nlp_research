from typing import List

import pandas as pd
from sklearn import preprocessing

import config


def read_corpus(filepath: str) -> (List[str], List[str]):
    sentences, tags = [], []
    sent, tag = ["<START>"], ["<START>"]
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            if line == "\n":
                if len(sent) > 1:
                    sentences.append(sent + ["<END>"])
                    tags.append(tag + ["<END>"])
                sent, tag = ["<START>"], ["<START>"]
            else:
                line = line.split()
                sent.append(line[0])
                tag.append(line[1])

    # enc_tag = preprocessing.LabelEncoder()
    # tags = enc_tag.fit_transform(tags)
    return sentences, tags


def process_csv(filepath: str) -> (List[str], List[str], preprocessing.LabelEncoder):
    enc_tag = preprocessing.LabelEncoder()
    df = (
        pd.read_csv(filepath, encoding="latin-1")
        .fillna(method="ffill")
        .drop(["POS"], axis=1)
    )
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tags = df.groupby("Sentence #")["Tag"].apply(list).values

    sentences = [["<START>"] + sent + ["<END>"] for sent in sentences]
    tags = [[0] + tag + [0] for tag in tags]
    return sentences, tags, enc_tag


if __name__ == "__main__":
    sentences, tags, enc_tag = process_csv(config.TRAIN_DATA)

    print(sentences[0])

    print(tags[0])

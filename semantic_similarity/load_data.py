import pandas as pd


def load_data():
    """
    loading and splitting of partitioned dataset for semantic similarity

    """

    train = "sts_b/sts-train.csv"
    test = "sts_b/sts-test.csv"
    dev = "sts_b/sts-dev.csv"

    stored = []
    for text_set in [("train", train), ("test", test), ("dev", dev)]:
        with open(text_set[1], "r") as f:
            data = f.read()
            for idx, line in enumerate(data.split("\n")):
                line = tuple(line.split("\t"))
                try:
                    stored.append(
                        {
                            "index": idx,
                            "partition": text_set[0],
                            "sentence_1": line[5],
                            "sentence_2": line[6],
                            "similarity": int(float(line[4]) > 3),
                        }
                    )
                except:
                    pass
    stored = pd.DataFrame(stored)
    return stored


if __name__ == "__main__":
    df = load_data()
    v = 10
    print(df.similarity.value_counts())

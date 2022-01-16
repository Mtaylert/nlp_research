import pandas as pd
import torch
import config




class TfidfDataset:
    def __init__(self, input_text, targets, tfidf_modeler):

        self.targets = targets
        self.input_text = list(tfidf_modeler.transform(input_text).toarray())

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, item):
        text = self.input_text[item]
        return {
            "ids": torch.tensor(text, dtype=torch.float),
            "targets": torch.tensor(int(self.targets[item]), dtype=torch.float),
        }





if __name__ == '__main__':
    # Assigning weights to words using TfidfVectorizer
    df = pd.read_csv('/Users/matthew/Documents/nlp_training/nlp_research/toxic_comments/data/train.csv')
    df = df.sample(frac=.5)
    df = df.reset_index(drop=True)
    tfidf_fit = config.TFIDF.fit(df['comment_text'])
    print(len(tfidf_fit.get_feature_names()))

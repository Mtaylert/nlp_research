import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer




class TfidfDataset:
    def __init__(self, input_text, targets, tfidf_modeler):

        self.targets = targets
        self.input_text = list(tfidf_modeler.transform(input_text).toarray())

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, item):
        text = self.input_text[item]
        return {
            "ids": torch.tensor(text, dtype=torch.long),
            "targets": torch.tensor(int(self.targets[item]), dtype=torch.long),
        }





if __name__ == '__main__':
    # Assigning weights to words using TfidfVectorizer
    tfidf_vec = TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))
    df = pd.read_csv('/Users/matthew/Documents/nlp_training/nlp_research/toxic_comments/data/train.csv')
    df = df.sample(frac=.5)
    df = df.reset_index(drop=True)
    tfidf_fit = tfidf_vec.fit(df['comment_text'])
    train_dataset = TfidfDataset(list(df['comment_text']), list(df['severe_toxic']),tfidf_fit)
    DL = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for b in DL:
        print(b)
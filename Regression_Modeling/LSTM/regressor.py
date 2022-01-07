import sys
import numpy as np

sys.path.append("../")
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import data_setup
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BiLSTMRegressor
from torch import optim


class LSTMRegressor:
    def __init__(self, vocab_size, hidden_dim, embedding_dim, epochs, fold, retrain=False):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = "saved_models/"
        self.fold = fold
        self.model = BiLSTMRegressor(
            input_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim, )
        if retrain:
            self.model.to(self.device)
        else:
            self.model.load_state_dict(
                torch.load("{}/{}_fold_MODEL.model".format(self.save_path,self.fold), map_location=self.device
                )
            )

    def fit(self, X, y):
        train_dataset = data_setup.RegressorDatasetup(X, y)
        train_dl = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        self.model.train()
        for epoch in range(self.epochs):
            for idx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
                data = batch["ids"]
                targets = batch["targets"]
                data = data.to(self.device)
                targets = targets.to(self.device)
                output = self.model(data)
                loss = nn.functional.mse_loss(output, targets.unsqueeze(-1))
                self.optimizer.zero_grad()  # reset the gradients
                loss.backward()
                # gradient descent
                self.optimizer.step()


        torch.save(
            self.model.state_dict(), "{}/{}_fold_MODEL.model".format(self.save_path,self.fold)
        )

    def predict(self, X):
        final_outputs = []
        self.model.eval()
        test_dataset = data_setup.RegressorDatasetup(X, np.ones(len(X)))
        test_dl = DataLoader(test_dataset, batch_size=1)

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
                data = batch["ids"]
                data = data.to(self.device)

                outputs = self.model(data)
                final_outputs.extend(outputs.cpu().detach().numpy().tolist()[0])
        return final_outputs


if __name__ == "__main__":

    df = pd.read_csv("../dataset/train.csv")
    df['severe_toxic'] = df.severe_toxic * 2
    df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)).astype(int)
    df['y'] = df['y'] / df['y'].max()
    df = df[['comment_text','y']]
    scoring_data = pd.read_csv('../dataset/comments_to_score.csv')

    config.BuildFolds(df, n_folds=config.N_FOLDS, frac1=config.FRAC1,
                       frac1_factor=config.FRAC1_FACTOR, save_path=config.FOLD_PATH).run()

    stored = {}

    for fld in range(config.N_FOLDS):
        print("\n\n")
        print(f' ****************************** FOLD: {fld} ******************************')
        fold_df = pd.read_csv(f'folds/df_fld{fld}.csv')
        X = fold_df['comment_text']
        y = fold_df['y']
        _, vocab = config.Tokenizer(X, max_length=config.MAX_LENGTH).build_vocab()
        VOCAB_SIZE = len(vocab)
        model = LSTMRegressor(
            vocab_size=VOCAB_SIZE,
            hidden_dim=config.HIDDEN_DIM,
            embedding_dim=config.EMBEDDING_DIM,
            epochs=1,
            fold=fld,
            retrain=True
        )
        model.fit(X, y)
        predictions = model.predict(scoring_data['text'])
        print(predictions)
        stored[fld] = predictions

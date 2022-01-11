import pandas as pd
import numpy as np
import config
import lstm
import embedder
import data_setup
from tqdm import tqdm
import torch



class ExoticModler:
    def __init__(self, epochs, embedding_matrix, input_size, lr, fld):
        self.epochs = epochs
        self.model = lstm.LSTM(embedding_matrix, input_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.lr = lr
        self.save_path = 'saved_models/'
        self.fld = fld

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def fit(self, X, y):

        train_dataloader = data_setup.ExoticModelSetup(X, y, trainer=True).get_sequences()

        param_lrs = [{'params': param, 'lr': self.lr} for param in self.model.parameters()]
        optimizer = torch.optim.Adam(param_lrs, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

        for epoch in range(self.epochs):
            self.model.train()
            for idx, data in tqdm(
                    enumerate(train_dataloader), total=len(train_dataloader)
            ):
                x_batch = data[:-1]
                y_batch = data[-1]
                y_pred = self.model(*x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        torch.save(
            self.model.state_dict(), "{}/{}_fold.model".format(self.save_path, self.fld)
        )

    def predict(self, X):
        final_outputs = []
        self.model.eval()
        val_dataloader = data_setup.ExoticModelSetup(X, np.ones(len(X)), trainer=False).get_sequences()

        with torch.no_grad():
            for idx, data in tqdm(
                    enumerate(val_dataloader), total=len(val_dataloader)
            ):
                x_batch = data[:-1]
                y_pred = self.model(*x_batch)
                final_outputs.extend(
                    torch.sigmoid(y_pred.cpu()).detach().numpy().tolist()[0]
                )
        return final_outputs

if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    df['severe_toxic'] = df.severe_toxic * 2
    df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)).astype(int)
    df['y'] = df['y'] / df['y'].max()
    df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
    cache_dict = embedder.Embedder(df['text']).cache
    modeler = ExoticModler(epochs=2, embedding_matrix=cache_dict['embedding_matrix'], input_size = cache_dict['max_features'], lr=0.001, fld=0)
    modeler.fit(df['text'],df['y'])



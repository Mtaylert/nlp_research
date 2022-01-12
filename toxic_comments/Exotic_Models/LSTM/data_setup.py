import torch
import pandas as pd
from keras.preprocessing import text
from keras.preprocessing import  sequence
from torch.utils.data import DataLoader, TensorDataset
import config


class ExoticModelSetup:
    def __init__(self, input_text, targets, trainer = True):
        self.input_text = input_text
        self.targets = targets
        self.tokenizer = text.Tokenizer()
        self.tokenizer.fit_on_texts(list(self.input_text))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = trainer

    def get_sequences(self):
        tokenized_input = self.tokenizer.texts_to_sequences(self.input_text)
        encoding = sequence.pad_sequences(tokenized_input, maxlen=config.MAX_LEN)
        x_train_torch = torch.tensor(encoding, dtype=torch.long).to(self.device)
        y_train_torch = torch.tensor(self.targets, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(x_train_torch, y_train_torch)
        dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True) if self.trainer else DataLoader(train_dataset, batch_size=1, shuffle=True)
        return dataloader


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    df['severe_toxic'] = df.severe_toxic * 2
    df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)).astype(int)
    df['y'] = df['y'] / df['y'].max()
    df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
    exotic_dataloader = ExoticModelSetup(df['text'], df['y']).get_sequences()
    for idx, data in enumerate(exotic_dataloader):
        x_batch = data[:-1]
        y_batch = data[-1]
        print(x_batch, y_batch)
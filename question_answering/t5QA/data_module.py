import pytorch_lightning as pl
import pandas as pd
import transformers
from torch.utils.data import Dataset, DataLoader
import config
import data_reader


class QADataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: transformers.DistilBertTokenizerFast = config.TOKENIZER,
        source_max_token_len: int = config.MAX_TOKEN_LEN,
        target_max_token_len: int = config.MAX_TARGET_LEN,
    ):

        self.data = data
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        source_encodings = self.tokenizer(
            data_row["question"],
            data_row["context"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            data_row["answer_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            question=data_row["question"],
            context=data_row["context"],
            answer_text=data_row["answer_text"],
            input_ids=source_encodings["input_ids"].flatten(),
            attention_mask=source_encodings["attention_mask"].flatten(),
            labels=labels.flatten(),
        )


class QADataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: transformers.DistilBertTokenizerFast = config.TOKENIZER,
        batch_size: int = config.BATCH_SIZE,
        source_max_token_len=config.MAX_TOKEN_LEN,
        target_max_token_len=config.MAX_TARGET_LEN,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self):

        self.train_dataset = QADataset(self.train_df)
        self.test_dataset = QADataset(self.test_df)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=True, num_workers=4)


if __name__ == "__main__":

    sample_dataset = QADataset(data_reader.unpack_data("data/train-v2.0.json"))
    for data in sample_dataset:
        print(data["question"])
        break

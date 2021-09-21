import pandas as pd
import transformers
import torch

import config
import data_reader

class BuildEncodings:

    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: transformers.DistilBertTokenizerFast = config.TOKENIZER

                 ):

        self.context = list(data['context'])
        self.question = list(data['question'])
        self.answers = list(data['answer_text'])
        self.answer_start = list(data['answer_start'])
        self.answer_end = list(data['answer_end'])
        self.tokenizer = tokenizer




    def encode(self):

        encodings = self.tokenizer(
            self.context,
            self.question,
            truncation=True,
            padding =True,
        max_length=512)

        encodings.update({"start_positions":self.answer_start,
                          "end_positions":self.answer_end})

        return encodings


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


if __name__ == '__main__':
    filepath = 'data/train-v2.0.json'
    train_output = data_reader.unpack_data(filepath)
    train_encodings = BuildEncodings(train_output)
    train_dataset = BuildDataset(train_encodings)



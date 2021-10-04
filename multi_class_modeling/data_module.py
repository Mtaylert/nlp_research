
import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class ExampleDataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)
        self.max_len = 256
        self.batch_size = 3


    def __len__(self):
        return len(self.text)

    def setup(self):
        encoded_data = self.tokenizer.batch_encode_plus(
            self.text,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt'
        )


        transformer_dataset = TensorDataset(encoded_data['input_ids'],
                                            encoded_data['attention_mask'],
                                            torch.tensor(self.target))


        data_loader =  DataLoader(transformer_dataset,
                              sampler=RandomSampler(transformer_dataset),
                              batch_size=self.batch_size)

        return data_loader





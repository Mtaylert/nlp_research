import pandas as pd
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf
import ssl
import numpy as np
from typing import List, Dict
import config
from tqdm import tqdm

class TextEmbed:
    def __init__(self, text_data: List[str]):
        self.text_data = text_data
        self.model = SentenceTransformer(config.TEXT_WEIGHTS)
        self.embedding_dict = self._get_embeds_()

    def __len__(self):
        return len(self.text)


    def _get_embeds_(self) -> Dict[int,np.array]:

        text_embed_dict = {}

        for idx, text in enumerate(tqdm(self.text_data)):
            text_embedding = self.model.encode(text, convert_to_tensor=True)
            text_embed_dict[idx] = text_embedding
        return text_embed_dict


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    text_module = TextEmbed(text_data=train['title'])
    print(text_module.embedding_dict[0])


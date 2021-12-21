import torch
from torch import nn
import transformers
import config


class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        self.bert_1 = transformers.BertModel.from_pretrained(config.BERT_MODEL)
        self.bert_2 = transformers.BertModel.from_pretrained(config.BERT_MODEL)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(768*2, 1)

    def foward(self, ids1, ids2, token_type_ids1,token_type_ids2, mask1, mask2):
        _, b1 = self.bert_1(ids = ids1, token_type_ids=token_type_ids1, attention_mask = mask1)
        _, b2 = self.bert_2(ids=ids2, token_type_ids=token_type_ids2, attention_mask=mask2)
        return b1, b2





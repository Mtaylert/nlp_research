import torch
from torch import nn
import transformers
import config


class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_MODEL)
        self.bert_drop = nn.Dropout(0.3)
        # use 1 because this is binary
        self.out = nn.Linear(768, 1)

    def forward(self, ids, token_type_ids, mask):
        out1, out2 = self.bert(
            ids, token_type_ids=token_type_ids, attention_mask=mask, return_dict=False
        )
        bert_output = self.bert_drop(out2)
        linear_output = self.out(bert_output)
        return linear_output

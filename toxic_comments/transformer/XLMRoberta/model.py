from torch import nn
import transformers
import torch

class XLMRoberta(nn.Module):
    def __init__(self):
        super(XLMRoberta, self).__init__()
        self.xlmroberta = transformers.XLMRobertaModel.from_pretrained("xlm-roberta-large", return_dict=False, num_labels=1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):

        o1, o2 = self.xlmroberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        dropout = self.dropout(o2)
        logits = self.classifier(dropout)
        return logits
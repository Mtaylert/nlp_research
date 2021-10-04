import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        # use 1 because this is binary
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        out1, out2 = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids,
            return_dict = False
        )

        bert_output = self.bert_drop(out2)
        linear_output = self.out(bert_output)
        return linear_output

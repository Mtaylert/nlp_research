import torch.nn as nn
import transformers


class BERTBaseUncased(nn.Module):
    def __init__(self, num_labels = 4):
        super(BERTBaseUncased, self).__init__()

        self.num_labels = num_labels
        self.bert = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels= self.num_labels,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
        self.bert_drop = nn.Dropout(0.3)

        self.out = nn.Linear(768, self.num_labels)

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

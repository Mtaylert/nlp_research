import torch.nn as nn
from TorchCRF import CRF
import config
import transformers


class BiLSTMCRF(nn.Module):

    def __init__(self, num_tags: int):

        super(BiLSTMCRF,self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.dropout_size = config.DROPOUT
        self.embed_size = config.EMBED_SIZE
        self.hidden_size = config.HIDDEN_SIZE

        self.dropout = nn.Dropout(self.dropout_size)

        self.LSTM = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.output_dim = self.embed_size * 2
        self.hidden2tag = nn.Linear(self.output_dim, num_tags)

        self.crf = CRF(num_tags, batch_first=True)


    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        sequence_output, _ = self.LSTM(sequence_output)

        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        loss = -1*self.crf(emissions, tags, mask=attention_mask.byte())
        return loss, emissions

    def predict(self, emissions, attention_mask):
        return self.crf.decode(emissions, attention_mask.byte())












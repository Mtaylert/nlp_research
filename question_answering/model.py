import torch

import config
import transformers
import torch.nn as nn


class DistilBERTQnA(nn.Module):
    def __init__(self):
        super(DistilBERTQnA, self).__init__()

        self.distilbert = transformers.DistilBertForQuestionAnswering.from_pretrained(
            config.DISTILBERT_PATH
        )

    def forward(self, input_ids, attention_mask, start_positions, end_positions):

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        return outputs

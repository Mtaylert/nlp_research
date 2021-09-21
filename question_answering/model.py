import torch

import config
import transformers
import torch.nn as nn


class DistilBERTQnA(nn.Module):
    def __init__(self):
        super(DistilBERTQnA, self).__init__()

        self.distilbert = transformers.DistilBertForQuestionAnswering.from_pretrained(config.DISTILBERT_PATH)

    def forward(self, input_ids, attention_mask, start_position, end_position):

        outputs = self.distilbert(
            input_ids,
            attention_mask,
            start_positions=start_position,
            end_positions=end_position
        )
        return outputs



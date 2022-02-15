import transformers
import torch
import config


class MLP(torch.nn.Module):

    def __init__(self, input_dim, output_dim,
                 num_hidden_lyr=2, dropout_prob=0.5, return_layer_outs=False,
                 hidden_channels=None, bn=False):

        super().__init__()
        self.out_dim = output_dim
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.return_layer_outs =  return_layer_outs
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels)!= num_hidden_lyr:
            raise ValueError('num of hidden layers should be same as length of hidden channels')

        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.act_name = 'relu'
        self.activation = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList(list(
            map(self.weight_init, [torch.nn.Linear(self.layer_channels[i], self.layer_channels[i +1])])
        ))

class ReviewsRegressor:
    super(ReviewsRegressor,self).__init__:
    self.roberta = transformers.RobertaModel.from_pretrained(config.MODEL)



class ToxicityModel(nn.Module):
    def __init__(self, checkpoint=params['model'], params=params):
        super(ToxicityModel, self).__init__()
        self.checkpoint = checkpoint
        self.bert = AutoModel.from_pretrained(checkpoint, return_dict=False)
        self.layer_norm = nn.LayerNorm(params['output_logits'])
        self.dropout = nn.Dropout(params['dropout'])
        self.dense = nn.Sequential(
            nn.Linear(params['output_logits'], 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(params['dropout']),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        preds = self.dense(pooled_output)
        return preds

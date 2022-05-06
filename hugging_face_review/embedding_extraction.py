from transformers import AutoTokenizer
from torch import nn
from transformers import AutoConfig


model_cpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_cpt)
text = "time flies like an arrow"

config = AutoConfig.from_pretrained(model_cpt)
inputs = tokenizer(text, return_tensors='pt',add_special_tokens=False)
token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
input_embeddings = token_embed(inputs.input_ids)
print(input_embeddings)
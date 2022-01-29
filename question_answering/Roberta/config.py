import transformers
import torch


EPOCHS = 2
TOKENIZER = transformers.RobertaTokenizerFast.from_pretrained("deepset/roberta-base-squad2")
MAX_LEN = 384
DOC_STRIDE = 128
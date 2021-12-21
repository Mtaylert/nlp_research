import transformers

EPOCHS = 3
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
MAX_LEN = 256

BERT_MODEL = 'bert-base-uncased'

TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
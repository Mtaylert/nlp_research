import transformers

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
BERT_PATH = "input/bert-base-uncased"
MODEL_PATH = "output/blstm_crf.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH, do_lower_case=True
)

TRAIN_DATA = "ner_datasets/GMB.csv"
DROPOUT = 0.3
EMBED_SIZE = 256
HIDDEN_SIZE = 256
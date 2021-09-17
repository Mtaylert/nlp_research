import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 10
VALID_BATCH_SIZE = 5
EPOCHS =2
BERT_PATH = "../input/bert-base-uncased"
MODEL_PATH = "sentiment_model.bin"
TRAINING_FILE = "../data/IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True)
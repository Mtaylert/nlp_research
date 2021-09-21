import transformers


MAX_TOKEN_LEN = 384
MAX_TARGET_LEN = 32
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS =3
DISTILBERT_PATH = "input/distilbert-base-uncased"
MODEL_PATH = "output/question_answer_FT.bin"
TOKENIZER = transformers.DistilBertTokenizerFast.from_pretrained(
    DISTILBERT_PATH,
    do_lower_case=True)
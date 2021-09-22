import transformers

MAX_TOKEN_LEN = 512
MAX_TARGET_LEN = 32
BATCH_SIZE = 8
EPOCHS = 2
T5_PATH = "input/t5-base"
MODEL_PATH = "output/"
TOKENIZER = transformers.T5TokenizerFast.from_pretrained(
    T5_PATH, do_lower_case=True
)

TRAIN_DATA = "data/train-v2.0.json"
VAL_DATA = "data/dev-v2.0.json"
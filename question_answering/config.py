import transformers


MAX_TOKEN_LEN = 396
MAX_TARGET_LEN = 32
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS =3
T5_PATH = "input/t5-base"
MODEL_PATH = "output/question_answer_SQUAD_model.bin"
TRAINING_FILE = "data/train-v2.0.json"
TOKENIZER = transformers.T5TokenizerFast.from_pretrained(
    T5_PATH,
    do_lower_case=True)
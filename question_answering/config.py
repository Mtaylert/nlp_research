import transformers


MAX_LEN = 512
TRAIN_BATCH_SIZE = 10
VALID_BATCH_SIZE = 5
EPOCHS =2
T5_PATH = "input/t5-base"
MODEL_PATH = "output/question_answer_SQUAD_model.bin"
TRAINING_FILE = "data/train-v2.0.json"
TOKENIZER = transformers.T5TokenizerFast.from_pretrained(
    T5_PATH,
    do_lower_case=True)
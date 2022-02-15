import transformers
import torch


MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 1
EPOCHS = 2
MODEL = 'dbmdz/bert-base-german-uncased'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
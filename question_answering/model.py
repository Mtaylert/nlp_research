import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import transformers
import config


class T5Base(nn.Module):
    def __init__(self):
        super(T5Base, self).__init__()

        self.t5 = transformers.T5ForConditionalGeneration.from_pretrained(config.T5_PATH)









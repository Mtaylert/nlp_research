import os

import tensorflow as tf
import tensorflow_hub as hub
from wav2vec2 import Wav2Vec2Config

config = Wav2Vec2Config()
print("TF version:", tf.__version__)

AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2
pretrained_layer = hub.KerasLayer("wav2vec2_1/", trainable=True)

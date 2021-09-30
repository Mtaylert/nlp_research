import pandas as pd
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf


train = pd.read_csv('data/train.csv')
model = SentenceTransformer('all-distilroberta-v1')

vision_model = tf.keras.applications.VGG16(
    include_top=False,
    weights='model_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
)
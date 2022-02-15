import pandas as pd
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf
import ssl
import numpy as np


ssl._create_default_https_context = ssl._create_unverified_context

train = pd.read_csv('data/train.csv')
model = SentenceTransformer('all-distilroberta-v1')

IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)

vision_model = tf.keras.applications.MobileNet(
    input_shape = (IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

def get_text_embedding(model,text):
    text_embedding = model.encode(text, convert_to_tensor=True)
    return text_embedding

def get_image_embedding(model,image):
    image = tf.keras.preprocessing.image.load_img(image, target_size=size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    img_embeddings = model(input_arr)
    meanImgEmb1 = np.mean(img_embeddings,axis=0)
    meanImgEmb2 = np.mean(meanImgEmb1, axis=0)
    meanImgEmb = np.mean(meanImgEmb2,axis=0)
    return meanImgEmb


def build_multi_modal_embeds(text,image):
    image_embedding = get_image_embedding(vision_model, f'data/train_images/{image}')
    text_embedding = get_text_embedding(model, text)
    concat_embedding = np.concatenate((text_embedding, image_embedding), axis=0)
    norm = np.linalg.norm(concat_embedding)
    cmb_embed_normal = concat_embedding/norm
    return cmb_embed_normal



multi_modal_embed = build_multi_modal_embeds(text=train['title'].iloc[0],image=train['image'].iloc[0])
print(multi_modal_embed)
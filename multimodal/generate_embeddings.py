import pandas as pd
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf
import ssl
import numpy as np
from typing import List, Dict
import config
from tqdm import tqdm


class TextEmbedder:
    def __init__(self, text_data: List[str]):
        self.text_data = text_data
        self.model = SentenceTransformer(config.TEXT_WEIGHTS)
        self.embedding_dict = self._get_embeds_()

    def __len__(self):
        return len(self.text)

    def _get_embeds_(self) -> Dict[int, np.array]:

        text_embed_dict = {}

        for idx, text in enumerate(tqdm(self.text_data)):
            text_embedding = self.model.encode(text, convert_to_tensor=True)
            text_embed_dict[idx] = text_embedding
        return text_embed_dict


class ImageEmbedder:
    def __init__(self, image_data: dict):

        ssl._create_default_https_context = ssl._create_unverified_context
        self.images = image_data
        self.model = tf.keras.applications.MobileNet(
            input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
            include_top=False,
            weights=config.IMAGE_WEIGHTS,
        )

        self.image_embedding_dict = self._get_embeds()

    def _get_embeds(self):

        image_embed_dict = {}

        for idx, image_file_name in enumerate(tqdm(self.images)):

            image = tf.keras.preprocessing.image.load_img(
                f"data/train_images/{image_file_name}",
                target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            )
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])
            img_embeddings = self.model(input_arr)
            meanImgEmb1 = np.mean(img_embeddings, axis=0)
            meanImgEmb2 = np.mean(meanImgEmb1, axis=0)
            meanImgEmb = np.mean(meanImgEmb2, axis=0)
            image_embed_dict[idx] = meanImgEmb

        return image_embed_dict


if __name__ == "__main__":

    train = pd.read_csv("data/train.csv")
    text_module = TextEmbedder(text_data=train["title"])
    image_module = ImageEmbedder(image_data=train["image"])
    print(image_module.image_embedding_dict[0])

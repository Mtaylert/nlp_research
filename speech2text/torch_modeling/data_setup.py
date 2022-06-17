import pandas as pd
import re
import json
from scipy.io import wavfile
import numpy as np
import config
import os

def remove_special_characters(text):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\]\[\(\)\&\|\£"]'
    text = re.sub(chars_to_ignore_regex, "", text).lower()
    text = re.sub("\d", "", text).lower()
    return text


def read_wave(path):
    _, data = wavfile.read(path)
    return data


def text_preprocessing(path):
    df = pd.read_csv(path, sep="|", header=None)
    df = df[[0, 1]]
    df.columns = ["ID", "Transcription"]
    files = os.listdir('/content/gdrive/MyDrive/Torch_SST/resampled_audio/')
    df["Transcription"] = df["Transcription"].apply(remove_special_characters)
    df["wav_file"] = df["ID"].apply(lambda x: "re_{}.wav".format(x))
    df=df[df['wav_file'].isin(files)]
    df["path"] = df["ID"].apply(lambda x: "/content/gdrive/MyDrive/Torch_SST/resampled_audio/re_{}.wav".format(x))
    return df[['ID','Transcription','path']]

def extract_all_chars(batch):
    all_text = " ".join(batch["Transcription"])
    vocab = list(set(all_text))
    vocab_list = list(vocab)
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open("outputs/vocab.json", "w") as f:
        json.dump(vocab_dict, f)

    return {"vocab": [vocab], "all_text": [all_text], "vocab_ids": vocab_dict}


class SSTDataset:
    def __init__(self, text, audio):
        self.audio = audio
        self.text = text
        self.processor = config.PROCESSOR

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text.iloc[item])
        audio = self.audio.iloc[item]

        processed_audio = self.processor(audio, sampling_rate=config.SAMPLING_RATE).input_values[0]
        with self.processor.as_target_processor():
            processed_text_labels = self.processor(text).input_ids

        return {
            "input_values": processed_audio,
            "labels": processed_text_labels
        }


if __name__ == "__main__":
    df = text_preprocessing("../text_transcription/metadata.csv")
    sst = SSTDataset(text=df['Transcription'], audio=df['wav_array'])
    for x in sst:
        print(x)


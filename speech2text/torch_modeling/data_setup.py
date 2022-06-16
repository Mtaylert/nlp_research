import pandas as pd
import re

def remove_special_characters(text):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    text = re.sub(chars_to_ignore_regex, '', text).lower()
    return text


def text_preprocessing(path):
    df = pd.read_csv(path, sep='|', header=None)
    df = df[[0,1]]
    df.columns = ['ID','Transcription']
    df['Transcription'] = df['Transcription'].apply(remove_special_characters)
    df['wav_file'] = df['ID'].apply(lambda x: x+".wav")
    return df


def extract_all_chars(batch):
  all_text = " ".join(batch["Transcription"])
  vocab = list(set(all_text))
  vocab_list = list(vocab)
  vocab_dict = {v: k for k, v in enumerate(vocab_list)}
  return {"vocab": [vocab], "all_text": [all_text],'vocab_ids':vocab_dict}


if __name__ == '__main__':
    df = text_preprocessing('../text_transcription/metadata.csv')
    vocab_data = extract_all_chars(df)
    print(vocab_data)
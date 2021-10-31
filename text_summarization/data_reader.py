import pandas as pd
import textwrap
import re
from gensim.models import FastText
import keras

data = pd.read_csv('BBC News Train.csv')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = text.replace('-','_')
    text = re.sub('\W+',' ', text)
    text = text.split()
    return text

def word_count(all_text):
    word_count_dict = {}
    for line in all_text:
        for word in line:
            if word in word_count_dict:
                word_count_dict[word]+=1
            else:
                word_count_dict[word] = 1
    vocab = word_count_dict.keys()
    return list(vocab), word_count_dict

def embed_text(sentences):
    model = FastText(vector_size=150, window=7, min_count=3, sg=1)
    model.build_vocab(data['text_clean'], update=False)  # Update the vocabulary
    model.train(data['text_clean'], total_examples=len(data['text_clean']), epochs=10)
    model.save('output/fasttext.model')


if __name__ == '__main__':
    data['text_clean']  = data['Text'].apply(preprocess)
    embed_text(data['text_clean'])

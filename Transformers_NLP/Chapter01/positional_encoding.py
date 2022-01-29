import torch
import nltk
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action = 'ignore')
nltk.download('punkt')


dprint=0 # prints outputs if set to 1, default=0

#‘text.txt’ file
sample = open("text.txt", "r")
s = sample.read()
# processing escape characters
f = s.replace("\n", " ")

data = []

# sentence parsing
for i in sent_tokenize(f):
	temp = []
	# tokenize the sentence into words
	for j in word_tokenize(i):
		temp.append(j.lower())
	data.append(temp)
model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 512,window = 5, sg = 1)

word1='black'
word2='brown'
pos1=2
pos2=10
a=model2.wv[word1]
b=model2.wv[word2]
print(a)
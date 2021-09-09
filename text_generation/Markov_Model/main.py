import pandas as pd
from text_preprocess import *
from Markov_Chain import MarkovChain
import numpy as np
data = pd.read_csv("../data/shortjokes.csv")
input_text = '. '.join(data['Joke'])

jokes_text = TextProcess(input_text)


mchain_model = MarkovChain(jokes_text, n = 3)

prefixes = ['the people are', 'remember to bring']
temperatures = [1, 0.7, 0.4, 0.1]

for temp in temperatures:
    print('temperature', temp)
    print(mchain_model.generate_sequence(np.random.choice(prefixes), 15 ,temperature=temp))
    print('\n')
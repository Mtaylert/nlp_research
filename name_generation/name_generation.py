import torch
from torch import nn
import string
import random
import unidecode


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_characters = string.printable
n_chars = len(all_characters)

file = unidecode.unidecode(open('names.txt').read())
print(len(file))
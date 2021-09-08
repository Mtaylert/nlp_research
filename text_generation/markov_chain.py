from scipy.sparse import csr_matrix
import numpy as np
import re

def preprocess(text, k = 2):
    text = re.sub("[^A-z,.!?'\n ]+","", text)
    text = re.sub("([.,!?])",r" \1 ", text)
    tokens = text.lower().split()
    distinct_states = list(set(tokens))
    return tokens,distinct_states

def k_preprocess(text, k = 2):
    text = re.sub("[^A-z,.!?'\n ]+", "", text)
    text = re.sub("([.,!?])", r" \1 ", text)
    tokens = text.lower().split()
    states = [tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]
    distinct_states = list(set(states))
    return distinct_states, states, tokens

def create_dense_matrix(distinct_states,tokens, k=2):
    #INIT
    m = csr_matrix(
        (len(distinct_states), len(distinct_states)),
        dtype=int
    )
    state_index = dict(
        [(state, row_num) for row_num, state in enumerate(distinct_states)]
    )

    for i in range(len(tokens) - k):
        state = tuple(tokens[i:i + k])
        next_state = tuple(tokens[i + 1:i + 1 + k])
        row = state_index[state]
        col = state_index[next_state]
        m[row, col] += 1

    return m, state_index


def generate_text(m, state_index, distinct_states, sentence_length=4):
    start_state_index = np.random.randint(len(distinct_states))
    state = distinct_states[start_state_index]
    output = ' '.join(state).capitalize()
    capitalize = False

    num_sentences = 0
    while num_sentences < sentence_length:
        row = m[state_index[state], :]
        probabilities = row / row.sum()

        # Sample next token from the probability distribution
        next_state_index = np.random.choice(
            len(distinct_states),
            1,
            p=probabilities.toarray()[0]
        )
        next_state = distinct_states[next_state_index[0]]

        # Punctuation and capitalization
        if next_state[-1] in ('.', '!', '?'):
            output += next_state[-1] + '\n\n'
            capitalize = True
            num_sentences += 1
        elif next_state[-1] == ',':
            output += next_state[-1]
        else:
            if capitalize:
                output += next_state[-1].capitalize()
                capitalize = False
            else:
                output += " " + next_state[-1]

        state = next_state
    return output



files = ['grimm_tales.txt','robin_hood_prologue.txt']
text = ""
for f in files:
    with open(f, 'r') as f:
        text += f.read()



distinct_states, states, tokens = k_preprocess(text, k=3)
m, state_index = create_dense_matrix(distinct_states,tokens,k=3)
output = generate_text(m, state_index, distinct_states, sentence_length=4)
print(output)

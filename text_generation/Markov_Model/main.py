from text_preprocess import TextProcess
from Markov_Chain import MarkovChain



if __name__ == '__main__':
    with open("../data/education_section_KAGGLE.txt") as f:
        data = f.read().split('\n')

    input_text = '. '.join(data)
    education_text = TextProcess(input_text)
    mchain_model = MarkovChain(education_text, n=3)
    prefixes = ['B.Sc in Maths','Bachelor of Engineering','Master of Science']

    for r in range(10):
        for x in prefixes:
            text = mchain_model.generate_sequence(x, 10, temperature=1)
            print(text)

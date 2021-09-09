import re
from typing import Dict, List


class TextProcess:
    def __init__(self, input_text: str):
        self.content = input_text
        self.tokens, self.tokens_distinct = self.tokenize()
        self.token2ind, self.ind2token = self.create_word_mapping(self.tokens_distinct)
        self.tokens_ind = [
            self.token2ind[token]
            if token in self.token2ind.keys()
            else self.token2ind["<| unknown |>"]
            for token in self.tokens
        ]

    def __repr__(self):
        return self.content

    def __len__(self):
        return len(self.tokens_distinct)

    @staticmethod
    def create_word_mapping(values_list: List) -> (Dict[str, int], Dict[int, str]):
        values_list.append("<| unknonwn |>")
        value2ind = {value: ind for ind, value in enumerate(values_list)}
        ind2value = dict(enumerate(values_list))
        return value2ind, ind2value

    def preprocess(self):
        punctuation_pad = "!?.,:-;"
        punctuation_remove = '"()_\n'
        self.content_preprocess = re.sub(r"(\S)(\n)(\S)", r"\1 \2 \3", self.content)

        # remove punctuation
        self.content_preprocess = self.content_preprocess.translate(
            str.maketrans("", "", punctuation_remove)
        )

        # add spacing after punctuation
        self.content_preprocess = self.content_preprocess.translate(
            str.maketrans({key: " {0} ".format(key) for key in punctuation_pad})
        )

        self.content_preprocess = re.sub(" +", " ", self.content_preprocess)
        self.content = self.content_preprocess.strip()

    def tokenize(self) -> (List, List):
        self.preprocess()
        tokens = self.content.split(" ")
        return tokens, list(set(tokens))

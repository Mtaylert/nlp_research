import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import config


def unpack_data(filepath: str) -> pd.DataFrame:
    with open(filepath) as f:
        data = json.loads(f.read())


    questions = data['data'][0]['paragraphs']

    data_rows = []
    for question in questions:
        context = question['context']
        for question_and_answer in question['qas']:
            question  = question_and_answer['question']
            answers = question_and_answer['answers']

            for answer in answers:
                answer_text = answer['text']
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer_text)

                data_rows.append({"question":question,
                                  "context":context,
                                  "answer_text":answer_text,
                                  "answer_start":answer_start,
                                  "answer_end":answer_end})

    return pd.DataFrame(data_rows)




class QADataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame):

        self.data = data
        self.tokenizer = config.TOKENIZER
        self.max_token_len = config.MAX_TOKEN_LEN
        self.max_target_len = config.MAX_TARGET_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        source_encoding = self.tokenizer(
        data_row['question'],
        data_row['context'],
         max_length=self.max_token_len,
         padding = 'max_length',
         truncation='only_second',
         return_attention_mask=True,
         add_special_tokens=True,
         return_tensors='pt'
                             )

        target_encoding = self.tokenizer(

            data_row['answer_text'],
            max_length=self.max_target_len,
            padding = 'max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'


        )

        labels = target_encoding['input_ids']
        labels[labels == 0] = -100

        cache = dict(
            question=data_row['question'],
            context = data_row['context'],
            answer_text = data_row['answer_text'],
            input_ids = source_encoding['input_ids'].flatten(),
            attention_mask = source_encoding['attention_mask'].flatten(),
            labels = labels.flatten()

        )
        return cache


if __name__ == '__main__':
    filepath = 'data/train-v2.0.json'
    train_output = unpack_data(filepath)

    sample_dataset = QADataset(train_output)
    for data in sample_dataset:
        print("Question: ", data['question'])
        print("Answer text: ", data['answer_text'])
        print("Input_ids: ", data['input_ids'][:10])
        print("Labels: ", data['labels'][:10])
        break

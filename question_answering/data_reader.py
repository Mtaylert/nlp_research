import json
import pandas as pd


def unpack_data(filepath: str) -> pd.DataFrame:
    with open(filepath) as f:
        data = json.loads(f.read())


    data_rows = []

    for group in data['data']:
      for question in group['paragraphs']:
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

    return pd.DataFrame(data_rows).sample(n=2500)


def add_end_idx(dataframe):
    # loop through each answer-context pair

    start_idx_save, end_idx_save = [] , []
    for answer, context, start_idx in zip(dataframe['answer_text'], dataframe['context'], dataframe['answer_start']):
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer
        # we already know the start index
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            end_idx_save.append(end_idx)
            start_idx_save.append(start_idx)
        else:
            # this means the answer is off by 1-2 tokens
            for n in [1, 2]:
                if context[start_idx - n:end_idx - n] == gold_text:

                    end_idx_save.append(end_idx - n)
                    start_idx_save.append(start_idx - n)

    dataframe['answer_start'] = start_idx_save
    dataframe['answer_end']  = end_idx_save
    return dataframe





if __name__ == '__main__':
    filepath = 'data/train-v2.0.json'
    train_output = unpack_data(filepath)

    print(train_output.columns)

import json
import pandas as pd

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

if __name__ == '__main__':
    filepath = 'data/train-v2.0.json'
    train_output = unpack_data(filepath)

    print(train_output.columns)

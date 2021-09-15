import json
import pandas as pd
from collections import defaultdict


def unpack_questions(nested_json):
    unpacked_dict = defaultdict()
    nested_json = nested_json.values[0][0]['qas']
    for line in nested_json:
        for key in line:
            if type(line[key]) == str:
                unpacked_dict[key] = line[key]

            elif type(line[key]) == list:
                if line[key]:
                    unpacked_dict[key] = line[key][0]['text']
    return dict(unpacked_dict)


def unpack_data(filepath):
    with open(filepath) as f:
        data = json.loads(f.read())

    question_sections = pd.io.json.json_normalize(data['data'])
    saved_output = []
    for section in question_sections['title']:
        current_section = unpack_questions(question_sections[question_sections['title']==section]['paragraphs'])
        saved_output.append(current_section)

    saved_output_df = pd.DataFrame(saved_output)
    saved_output_df['answers']=saved_output_df['answers'].fillna(saved_output_df['plausible_answers'])
    return saved_output_df[['question','answers']]



if __name__ == '__main__':
    filepath = 'data/train-v2.0.json'
    train_output = unpack_data(filepath)
    print(train_output)


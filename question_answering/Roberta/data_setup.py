import pandas as pd
import transformers
import torch
import config

class QnADataset:
    def __init__(self, text_features, answer, input_type='train'):
        self.text_features = text_features
        self.input_type = input_type

    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, item):
        if self.input_type =='triain':
            self.question = list(self.text_features['question'])[item]
            self.context = list(self.text_features['context'])[item]
            self.answer = list(self.text_features['answer'])[item]
            self.start_position = list(self.text_features['start_position'])[item]
            self.end_position = list(self.text_features['end_position'])[item]
            self.tokenized_question = config.TOKENIZER.encode_plus(
                self.question,
                self.context,
                truncation="only_second",
                max_length=config.MAX_LEN,
                stride=config.DOC_STRIDE,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt"
            )

            self.tokenized_answer = config.TOKENIZER.encode_plus(self.answer,
                                                                 max_length=config.MAX_LEN,
                                                                 padding="max_length",
                                                                 truncation=True,
                                                                 return_attention_mask=True,
                                                                 add_special_tokens=True,
                                                                 return_tensors="pt",
            )

            return {
                'input_ids':torch.tensor(self.tokenized_question['input_ids'], dtype=torch.long),
                'attention_mask':torch.tensor(self.tokenized_question['attention_mask'],dtype=torch.long),
                'offset_mapping':torch.tensor(self.tokenized_question['offset_mapping'], dtype=torch.long),
                'start_position':torch.tensor(self)


            }

            self.answer_ids = torch.tensor(self.tokenized_answer['input_ids'], dtype=torch.long)








if __name__ == '__main__':
    df = pd.read_csv('../datasets/train.csv')
    print(df.head())
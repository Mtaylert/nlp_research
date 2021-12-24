import torch
import config

class BertDatasetTraining:
    def __init__(self, input1, input2, target):
        self.input1 = input1
        self.input2 = input2
        self.target = target

    def __len__(self):
        return len(self.input1)


    def __getitem__(self, item):
        input1 = str(self.input1[item])
        input2 = str(self.input2[item])

        input1 = " ".join(input1.split())
        input2 = " ".join(input2.split())


        inputs = config.TOKENIZER.encode_plus(input1, input2,
                                              add_special_tokens=True,
                                              max_length=config.MAX_LEN,
                                              padding='max_length',
                                              )

        ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        mask = inputs['attention_mask']


        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(int(self.target[item]), dtype=torch.long)
        }




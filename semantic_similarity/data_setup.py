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


        first_input = config.TOKENIZER.encode_plus(input1, None,
                                              add_special_tokens=True,
                                              max_length=config.MAX_LEN,
                                              pad_to_max_length=True,
                                              )
        second_input = config.TOKENIZER.encode_plus(input2, None,
                                              add_special_tokens=True,
                                              max_length=config.MAX_LEN,
                                              pad_to_max_length=True,
                                              )
        ids1 = first_input['input_ids']
        token_type_ids1 = first_input['token_type_ids']
        mask1 = first_input['attention_mask']

        ids2 = second_input['input_ids']
        token_type_ids2 = second_input['token_type_ids']
        mask2 = second_input['attention_mask']

        return {
            "ids1": torch.tensor(ids1, dtype=torch.long),
            "mask1": torch.tensor(mask1, dtype=torch.long),
            "token_type_ids1": torch.tensor(token_type_ids1, dtype=torch.long),
            "ids2": torch.tensor(ids2, dtype=torch.long),
            "mask2": torch.tensor(mask2, dtype=torch.long),
            "token_type_ids2": torch.tensor(token_type_ids2, dtype=torch.long),
            "targets": torch.tensor(int(self.target[item]), dtype=torch.long)
        }



        }
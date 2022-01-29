import sys
from load_data import load_data
import transformers
import tensorflow as tf
from transformers import TFBertForSequenceClassification


class TFDatasetup:
    def __init__(self, input_df):
        self.text_input_1 = input_df['sentence_1']
        self.text_input_2 = input_df['sentence_2']
        self.target = input_df['similarity']
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def __len__(self):
        return len(self.text_input_1)

    def __getitem__(self, item):
        text1 = list(self.text_input_1)[item]
        text2 = list(self.text_input_2)[item]

        inputs = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=384,
            padding="max_length",
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "input_ids": ids,
            "token_type_ids": token_type_ids,
            "attention_mask": mask,
            "target": self.target[item],
        }

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks,
           }, label


def ecode_data(df):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    datasetup = TFDatasetup(train)
    for bert_input in datasetup:

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([bert_input['target']])
    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)


if __name__ == '__main__':
    df = load_data()

    train = df[df["partition"] == "train"]
    train = train.sample(n=200)
    train = train.reset_index(drop=True)

    dev = df[df["partition"] != "train"].reset_index(drop=True)
    dev = dev.iloc[0:20]

    train_setup = ecode_data(train)
    test_setup = ecode_data(dev)


    batch_size = 6
    learning_rate = 2e-5

    # we will do just 1 epoch for illustration, though multiple epochs might be better as long as we will not overfit the model
    number_of_epochs = 1

    # model initialization
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

    # classifier Adam recommended
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

    # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    bert_history = model.fit(train_setup, epochs=number_of_epochs, validation_data=test_setup)
    predictions = model.predict(test_setup)
    print(predictions)

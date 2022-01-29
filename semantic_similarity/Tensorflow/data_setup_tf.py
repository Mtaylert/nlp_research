from transformers import TFBertForSequenceClassification
import tensorflow as tf
import tensorflow_datasets as tfds
(ds_train, ds_test), ds_info = tfds.load('imdb_reviews',
          split = (tfds.Split.TRAIN, tfds.Split.TEST),
          as_supervised=True,
          with_info=True)

print(ds_train.shape)
'''
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def convert_example_to_feature(review):
    # combine step for tokenization, WordPiece vector mapping and will add also special tokens and truncate reviews longer than our max length

    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to BERT
                                 pad_to_max_length=True,  # add [PAD] tokens
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 )


def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks,
           }, label


def encode_examples(ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)

    for review, label in tfds.as_numpy(ds):
        bert_input = convert_example_to_feature(review.decode())

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)


# can be up to 512 for BERT
max_length = 512

# the recommended batches size for BERT are 16,32 ... however on this dataset we are overfitting quite fast
# and smaller batches work like a regularization.
# You might play with adding another dropout layer instead.

batch_size = 6

learning_rate = 2e-5

# we will do just 1 epoch for illustration, though multiple epochs might be better as long as we will not overfit the model
number_of_epochs = 1


# model initialization
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# classifier Adam recommended
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# train dataset
ds_train_encoded = encode_examples(ds_train).shuffle(200).batch(batch_size)

# test dataset
ds_test_encoded = encode_examples(ds_test).batch(batch_size)
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)
'''
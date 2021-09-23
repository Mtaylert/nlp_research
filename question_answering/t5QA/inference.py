import pandas as pd

from model import QAModel
import config
import data_reader

def generate_answer(question):
    trained_model = QAModel.load_from_checkpoint('output/best-checkpoint_QA.ckpt')
    trained_model.freeze()
    tokenizer = config.TOKENIZER
    source_encoding = tokenizer(question['question'],
                                question['context'],
                                max_length=512,
                                padding=True,
                                truncation='only_second',
                                return_attention_mask=True,
                                add_special_tokens=True,
                                return_tensors='pt'
                                )

    generated_ids  = trained_model.model.generate(
        input_ids = source_encoding['input_ids'],
        attention_mask = source_encoding['attention_mask'],
        num_beams = 1,
        max_length=80,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True

    )
    preds = [tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
             for generated_id in generated_ids


             ]

    return "".join(preds)



if __name__ =='__main__':

    val_df = data_reader.unpack_data('data/dev-v2.0.json')
    example_question = pd.DataFrame.from_dict({"question":'What is the capital of Virginia?',
                                     "context":"This state capital of Richmond has 225000 people living in it.",
                                     "answer":'Richmond',
                                     "answer_start":0,
                                     "answer_end":10},orient='index').T



    print(generate_answer(example_question.iloc[0]))


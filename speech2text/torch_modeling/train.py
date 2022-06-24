from datasets import load_dataset, load_metric
import config
import modeling
import data_setup
import numpy as np
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.model_selection import train_test_split

def compute_metrics(pred):
    wer_metric = load_metric("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = config.PROCESSOR.tokenizer.pad_token_id

    pred_str = config.PROCESSOR.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = config.PROCESSOR.batch_decode(pred.label_ids, group_tokens=False)
    print(pred_str)
    print(label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def run():
    data_collator = modeling.DataCollatorCTCWithPadding(processor=config.PROCESSOR, padding=True, max_length_labels=32)
    df = data_setup.text_preprocessing("../text_transcription/metadata.csv")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=101)

    train_df = train_df[["path", "Transcription"]]
    train_df = train_df.reset_index(drop=True)

    test_df = test_df[["path", "Transcription"]]
    test_df = test_df.reset_index(drop=True)
    train_df.to_csv("outputs/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv("outputs/test.csv", sep="\t", encoding="utf-8", index=False)
    common_voice_train = load_dataset("csv", data_files={"train": "outputs/train.csv"}, delimiter="\t")["train"]
    common_voice_test = load_dataset("csv", data_files={"test": "outputs/test.csv"}, delimiter="\t")["test"]
    print(common_voice_test)

if __name__ == '__main__':
    run()


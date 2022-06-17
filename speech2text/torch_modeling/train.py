from datasets import load_dataset, load_metric
import config
import modeling
import data_setup
import numpy as np
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer


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
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=config.PROCESSOR.tokenizer.pad_token_id,
    )
    model.freeze_feature_extractor()
    training_args = TrainingArguments(
        output_dir='outputs/',
        group_by_length=True,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=2,
        fp16=False,
        gradient_checkpointing=True,
        save_steps=50,
        eval_steps=50,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        save_total_limit=2,
        disable_tqdm=False,
    )
    data_collator = modeling.DataCollatorCTCWithPadding(processor=config.PROCESSOR, padding=True, max_length_labels=32)
    df = data_setup.text_preprocessing("../text_transcription/metadata.csv")
    sst = data_setup.SSTDataset(text=df['Transcription'], audio=df['wav_array'])
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=sst,
        eval_dataset=sst,
        tokenizer=config.PROCESSOR.feature_extractor,
    )

    trainer.train()


if __name__ == '__main__':
    run()


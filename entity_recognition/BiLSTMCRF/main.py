from typing import BinaryIO

import torch
import transformers
from sklearn import model_selection
from tqdm import tqdm

import config
import data_module
import data_reader
from model import BiLSTMCRF


def train(save: bool = False) -> dict[str, str]:

    sentences, tags, enc = data_reader.process_csv(config.TRAIN_DATA)
    enc_classes = len(enc.classes_)

    train_sent, val_sent, train_tag, val_tag = model_selection.train_test_split(
        sentences, tags, test_size=0.2, random_state=42
    )

    train_dataset = data_module.BLSTMCRFDataset(text=train_sent, target=train_tag)
    val_dataset = data_module.BLSTMCRFDataset(text=val_sent, target=val_tag)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMCRF(num_tags=enc_classes)
    model.to(device)
    num_train_steps = int(len(train_sent) / config.BATCH_SIZE * config.EPOCHS)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=3e-5)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    model.train()

    loop = tqdm(train_dataloader)
    for ep in range(config.EPOCHS):
        for step, batch in enumerate(loop):
            input_ids = batch["ids"]
            mask = batch["mask"]
            token_type_ids = batch["token_type_ids"]
            targets = batch["targets"]

            input_ids = input_ids.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            loss, emissions = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=mask,
                tags=targets,
            )
            crf_pred = model.predict(emissions, mask)

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            final_loss += loss.item()
    if save:
        torch.save(model.state_dict(), config.MODEL_PATH)

    model.eval()
    final_loss = 0
    val_loop = tqdm(val_dataloader)
    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for step, batch in enumerate(val_loop):
            input_ids = batch["ids"]
            mask = batch["mask"]
            token_type_ids = batch["token_type_ids"]
            targets = batch["targets"]

            input_ids = input_ids.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            targets = targets.to(device)

            loss, emissions = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=mask,
                tags=targets,
            )
            final_loss += loss.item()

            crf_pred = model.predict(emissions, mask)

            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_outputs.extend(crf_pred)

    cache = {
        "final_outputs": final_outputs,
        "final_targets": final_targets,
        "test_data": val_sent,
        "test_tags": val_tag,
        "encoder": enc,
        "data_module": val_dataloader,
    }

    return cache


if __name__ == "__main__":
    train()

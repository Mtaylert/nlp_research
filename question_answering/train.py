from torch.utils.data import DataLoader
from transformers import AdamW
import torch
from tqdm import tqdm

from model import DistilBERTQnA
import dataset
import data_reader
import config


def run():

    train_path = "data/train-v2.0.json"
    valid_path = "data/dev-v2.0.json"
    train_df = data_reader.unpack_data(train_path)
    val_df = data_reader.unpack_data(valid_path)

    train_dataset = dataset.BuildDataset(dataset.BuildEncodings(train_df).encode())
    val_dataset = dataset.BuildDataset(dataset.BuildEncodings(val_df).encode())

    train_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.VALID_BATCH_SIZE)

    model = DistilBERTQnA()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(config.EPOCHS):
        loop = tqdm(train_loader)
        for batch in loop:
            optim.zero_grad()

            # INPUTS
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # TARGETS
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            loss = outputs[0]

            loss.backward()
            optim.step()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), config.MODEL_PATH)

    print("DONE FINE TUNINING")

    model.eval()
    acc = []

    loop = tqdm(val_loader)
    for batch in loop:
        with torch.no_grad():
            # INPUTS
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # TARGETS
            start_true = batch["start_positions"].to(device)
            end_true = batch["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_true,
                end_positions=end_true,
            )

            start_pred = torch.argmax(outputs["start_logits"], dim=1)
            end_pred = torch.argmax(outputs["end_logits"], dim=1)

            acc.append(((start_pred == start_true).sum() / len(start_pred)).item())

            acc.append(((end_pred == end_true).sum() / len(end_pred)).item())

    print(sum(acc) / len(acc))


if __name__ == "__main__":
    run()

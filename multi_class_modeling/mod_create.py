import data_module
import main
import config
import transformers
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics


def train(train_text, train_labels, save_path):
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    train_text = [main.remove_stopwords(text_line) for text_line in train_text]
    train_dataloader = data_module.ExampleDataset(text=train_text, target=train_labels, train_flag=True).setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                       num_labels=config.CLASS_SIZE,
                                                                       output_attentions=False,
                                                                       output_hidden_states=False)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_steps = int(len(train_text) / config.BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_train_steps)
    top_accuracy = 0
    for epoch in range(1, config.EPOCHS + 1):

        batch_predictions, batch_actuals = [], []
        model.train()

        loss_train_total = 0

        progress_bar = tqdm(train_dataloader)
        for batch in progress_bar:
            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1].detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            for pred, act in zip(np.argmax(logits, axis=1), label_ids):
                batch_predictions.append(pred)
                batch_actuals.append(act)

            loss_train_total += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        batch_acc = metrics.accuracy_score(batch_actuals, batch_predictions)
        if batch_acc > top_accuracy:
            print("Current model is better than previous {} > {}".format(batch_acc, top_accuracy))
            torch.save(model.state_dict(), '{}/BEST_MODEL.model'.format(save_path))

    torch.save(model.state_dict(), '{}/LAST_MODEL.model'.format(save_path))


def predict(val_text, val_labels, model_filepath):
    val_text = [main.remove_stopwords(text_line) for text_line in val_text]

    val_dataloader = data_module.ExampleDataset(text=val_text, target=val_labels, train_flag=True).setup()

    model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                       num_labels=config.CLASS_SIZE,
                                                                       output_attentions=False,
                                                                       output_hidden_states=False)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in val_dataloader:
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return predictions, true_vals







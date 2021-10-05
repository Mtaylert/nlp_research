import data_module
import main
import transformers
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from tqdm import tqdm




def fit(filepath,save_path):
    bitch_size =32
    epochs = 2
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    X_train, X_val, y_train, y_val, enc_tag = main.read_dataset(filepath)

    train_dataloader= data_module.ExampleDataset(text=X_train, target=y_train, train_flag=True).setup()
    val_dataloader = data_module.ExampleDataset(text=X_val, target=y_val, train_flag=True).setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=len(enc_tag.classes_),
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_steps = int(len(X_train) / bitch_size * epochs)

    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_train_steps)

    for epoch in range(1, epochs + 1):

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
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()



    torch.save(model.state_dict(), '{}/finetuned_BERT.model'.format(save_path))
    class_size = len(enc_tag.classes_)

    return val_dataloader, class_size


def predict(val_dataloader,class_size,model_filepath):


    model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=class_size,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in val_dataloader:
        batch = tuple(b.to(device) for b in batch)

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

if __name__ == '__main__':
    val_dataloader = fit(filepath='data.csv',
                                    save_path='fine_tuned_models/')

    preds, true_labels = predict(val_dataloader,
                                            model_filepath='fine_tuned_models/finetuned_BERT.model')





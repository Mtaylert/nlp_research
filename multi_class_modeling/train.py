import data_module
import main
import transformers
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from tqdm import tqdm




def fit(filepath):
    bitch_size =32
    epochs = 2
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    X_train, X_val, y_train, y_val, enc_tag = main.read_dataset('data.csv')

    train_dataloader= data_module.ExampleDataset(text=X_train, target=y_train, train_flag=True).setup()

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



        torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')


if __name__ == '__main__':
    fit('data.csv')





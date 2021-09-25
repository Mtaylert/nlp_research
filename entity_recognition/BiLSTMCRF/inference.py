import numpy as np
import torch

import config
import data_module
import data_reader
from model import BiLSTMCRF


def infer_input(input_sentence: str):

    enc = data_reader.process_csv(config.TRAIN_DATA)[-1]

    tokenizer = config.TOKENIZER
    model = BiLSTMCRF(num_tags=len(enc.classes_))

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=map_location))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenized_sentence = config.TOKENIZER.encode(input_sentence)
    sentence = input_sentence.split()
    example_dataset = data_module.BLSTMCRFDataset(
        text=[sentence], target=[[0] * len(sentence)]
    )

    with torch.no_grad():
        data = example_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)

    loss, emissions = model(
        input_ids=data["ids"],
        attention_mask=data["mask"],
        token_type_ids=data["token_type_ids"],
        tags=data["targets"],
    )
    probability = (
        emissions.cpu().detach().numpy().tolist()[0][: len(tokenized_sentence)]
    )

    crf_pred = model.predict(emissions, data["mask"])
    predictions = enc.inverse_transform(crf_pred[0])[: len(tokenized_sentence)]
    position = np.argmax(probability, axis=1)
    max_probabilities = [probability[idx][pos] for idx, pos in enumerate(position)]

    token_preds = []
    for idx, (encoded_word_piece, pred) in enumerate(
        zip(tokenized_sentence[1:-1], predictions[1:-1])
    ):
        decoded_word_piece = tokenizer.decode(encoded_word_piece)
        if decoded_word_piece[0] != "#":
            token_preds.append(pred)

    cache = {
        "sentence": input_sentence,
        "max_probabilities": max_probabilities,
        "classes": enc.classes_,
        "predictions": token_preds,
        "probabilities_all": probability,
    }
    return cache


if __name__ == "__main__":
    sentence = "My name is Matt Taylert and I am from Richmond Virginia"

    output_cache = infer_input(sentence)

    print(output_cache["predictions"])

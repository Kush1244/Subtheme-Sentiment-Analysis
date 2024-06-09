import torch
import pickle
import configuration as config
import numpy as np
from model import SentimentClassifier
from transformers import (
    ElectraForSequenceClassification,
    ElectraTokenizer,
    ElectraConfig,
)
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--text", required=True, help="Input text")
args = vars(ap.parse_args())

device = config.device
model_config = ElectraConfig()

tokenizer = config.TOKENIZER
model = SentimentClassifier(config.electra_model, config.NUM_LABELS).to(config.device)
checkpoint = torch.load(config.MODEL_PATH)
model.load_state_dict(checkpoint)
encoder = open(config.ENCODER_PATH, "rb")
le = pickle.load(
    encoder
)  # This is the same encode used to encode the label during data preprocessing phase
encoder.close()
print("done")


# functions that returns labels given texts at inference, it takes text, model and tokenizer as arguments
def inference(text, model, tokenizer):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
    )
    ids = (
        torch.tensor(inputs["input_ids"], dtype=torch.long)
        .unsqueeze(0)
        .to(device, dtype=torch.long)
    )
    mask = (
        torch.tensor(inputs["attention_mask"], dtype=torch.long)
        .unsqueeze(0)
        .to(device, dtype=torch.long)
    )
    token_type_ids = (
        torch.tensor(inputs["token_type_ids"], dtype=torch.long)
        .unsqueeze(0)
        .to(device, dtype=torch.long)
    )

    # adding sigmoid layer at the end of model
    outputs = model(ids, mask, token_type_ids).sigmoid()

    prediction = [1 if i > 0.35 else 0 for i in outputs[0]]

    # extracting labels from the multilabel encoder
    labels = le.inverse_transform(np.array([prediction]))[0]
    print("Labels -- {}".format(list(labels)))
    return list(labels)


if __name__ == "__main__":
    labels = inference(args["text"], model, tokenizer)

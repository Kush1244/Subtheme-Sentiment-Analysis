import torch
import numpy as np
import pickle
import configuration as config
from transformers import BertTokenizer, BertConfig
from model import SentimentMultilabel, SentimentMultilabelLarge
from data import get_loader
from sklearn import metrics

device = config.device
model_config = BertConfig()
num_labels = config.NUM_LABELS
lr = config.LEARNING_RATE
epochs = config.EPOCHS

# evaluation metrics data, for analysis of models performance
eval_metrics = {
    "epochs": [],
    "train_loss": [],
    "val_loss": [],
    "training_f1_micro": [],
    "training_f1_macro": [],
    "val_f1_micro": [],
    "val_f1_macro": [],
    "training_hamming_loss": [],
    "val_hamming_loss": [],
}


# Binary cross entropy with logits loss function
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


eval_metrics = {
    "epochs": [],
    "train_loss": [],
    "val_loss": [],
    "training_f1_micro": [],
    "training_f1_macro": [],
    "val_f1_micro": [],
    "val_f1_macro": [],
    "training_hamming_loss": [],
    "val_hamming_loss": [],
}


# Function to take true and predicted labels and calculate and print multiple metrics
def print_metrics(true, pred, loss, type):
    pred = np.array(pred) >= 0.35
    hamming_loss = metrics.hamming_loss(true, pred)
    precision_micro = metrics.precision_score(
        true, pred, average="micro", zero_division=1
    )
    recall_micro = metrics.recall_score(true, pred, average="micro", zero_division=1)
    precision_macro = metrics.precision_score(
        true, pred, average="macro", zero_division=1
    )
    recall_macro = metrics.recall_score(true, pred, average="macro", zero_division=1)
    f1_score_micro = metrics.f1_score(true, pred, average="micro", zero_division=1)
    f1_score_macro = metrics.f1_score(true, pred, average="macro", zero_division=1)
    print("-------{} Evaluation--------".format(type))
    print("BCE Loss: {:.4f}".format(loss))
    print("Hamming Loss: {:.4f}".format(hamming_loss))
    print(
        "Precision Micro: {:.4f}, Recall Micro: {:.4f}, F1-measure Micro: {:.4f}".format(
            precision_micro, recall_micro, f1_score_micro
        )
    )
    print(
        "Precision Macro: {:.4f}, Recall Macro: {:.4f}, F1-measure Macro: {:.4f}".format(
            precision_macro, recall_macro, f1_score_macro
        )
    )
    print("------------------------------------")
    return f1_score_micro, f1_score_macro, hamming_loss, loss


# function to validate the validation data from trained model
def validate(model, testLoader):
    model.eval()
    val_targets = []
    val_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testLoader):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            epoch_loss = loss.item()
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        return print_metrics(val_targets, val_outputs, epoch_loss, "Validation")


# function to train the model
def train():
    model = (
        SentimentMultilabel(num_labels, model_config).to(device)
        if config.model == "base"
        else SentimentMultilabelLarge(num_labels, model_config).to(device)
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # creating the training and validation data loaders
    trainLoader, testLoader, _ = get_loader("output/")
    for epoch in range(1, epochs + 1):
        eval_metrics["epochs"].append(epoch)
        model.train()
        epoch_loss = 0
        # training actual and prediction for each epoch for printing metrics
        train_targets = []
        train_outputs = []
        for _, data in enumerate(trainLoader):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            epoch_loss = loss.item()
            train_targets.extend(targets.cpu().detach().numpy().tolist())
            train_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            if _ % 50 == 0:
                print(f"Epoch: {epoch}, Loss:  {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculating the evaluation scores for both training and validation data
        train_f1_micro, train_f1_macro, train_hamming, train_loss = print_metrics(
            train_targets, train_outputs, epoch_loss, "Training"
        )
        val_f1_micro, val_f1_macro, val_hamming, val_loss = validate(model, testLoader)
        eval_metrics["training_f1_micro"].append(train_f1_micro)
        eval_metrics["training_f1_macro"].append(train_f1_macro)
        eval_metrics["training_hamming_loss"].append(train_hamming)
        eval_metrics["val_f1_micro"].append(val_f1_micro)
        eval_metrics["val_f1_macro"].append(val_f1_macro)
        eval_metrics["val_hamming_loss"].append(val_hamming)
        eval_metrics["train_loss"].append(train_loss)
        eval_metrics["val_loss"].append(val_loss)

    # saving the metrics and trained model for inference and model analysis
    torch.save(model.state_dict(), config.BERT_MODEL_PATH)
    return True


if __name__ == "__main__":
    train()

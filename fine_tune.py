import torch
import numpy as np
from model import SentimentClassifier
import configuration as cfg
from data import get_loader
from sklearn import metrics
import pickle

model = SentimentClassifier(cfg.electra_model, cfg.NUM_LABELS).to(cfg.device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.LEARNING_RATE)
trainLoader, testLoader, _ = get_loader("output/")
device = cfg.device


def loss_fn(y, y_hat):
    return torch.nn.BCEWithLogitsLoss(y, y_hat)


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


def train():

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        epochLoss = 0
        trainTargets = list()
        trainOutputs = list()
        for i, data in enumerate(trainLoader):
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            y = data["targets"].to(device)
            y_hat = model(ids, mask, token_type_ids)
            optimizer.zero_grad()
            loss = loss_fn(y_hat, y)
            epochLoss = loss.item()
            trainTargets.extend(y.cpu().detach().numpy().tolist())
            trainOutputs.extend(torch.sigmoid(y_hat).cpu().detach().numpy().tolist())

            if i % 10 == 0:
                print(f"Epoch:{epoch} : Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_f1_micro, train_f1_macro, train_hamming, train_loss = print_metrics(
                trainTargets, trainOutputs, epochLoss, "Training"
            )
            val_f1_micro, val_f1_macro, val_hamming, val_loss = validate(
                model, testLoader
            )
            eval_metrics["training_f1_micro"].append(train_f1_micro)
            eval_metrics["training_f1_macro"].append(train_f1_macro)
            eval_metrics["training_hamming_loss"].append(train_hamming)
            eval_metrics["val_f1_micro"].append(val_f1_micro)
            eval_metrics["val_f1_macro"].append(val_f1_macro)
            eval_metrics["val_hamming_loss"].append(val_hamming)
            eval_metrics["train_loss"].append(train_loss)
            eval_metrics["val_loss"].append(val_loss)

    filehandler = open(b"output/bert_metrics.pkl", "wb")
    pickle.dump(eval_metrics, filehandler)
    filehandler.close()

    torch.save(model.state_dict(), cfg.MODEL_PATH)


if __name__ == "__main__":
    train()

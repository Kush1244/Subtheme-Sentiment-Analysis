import configuration
import torch
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset


# Dataset for finetuning Google's Electra Model
class SentimentDataset(Dataset):
    def __init__(self, df_path, tokenizer, max_len):
        self.df = pd.read_pickle(df_path)
        self.texts = self.df.text
        self.targets = self.df.encoded
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        text = self.texts[index]
        target = self.targets[index]
        # encoding the texts with pretrained Electra tokenizer
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(target, dtype=torch.long),
        }


# create datasets from preprocessed train and test files and then return data loaders for both
def get_loader(rootPath, train_batch_size=32, test_batch_size=8, shuffle=True):
    trainDataset = SentimentDataset(
        rootPath + "train.pkl", configuration.TOKENIZER, configuration.MAX_LEN
    )
    testDataset = SentimentDataset(
        rootPath + "test.pkl", configuration.TOKENIZER, configuration.MAX_LEN
    )

    trainLoader = DataLoader(
        dataset=trainDataset, batch_size=train_batch_size, shuffle=shuffle
    )

    testLoader = DataLoader(
        dataset=testDataset, batch_size=test_batch_size, shuffle=shuffle
    )

    return trainLoader, testLoader, configuration.TOKENIZER

import torch
import torch.nn as nn
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from dataclasses import dataclass


@dataclass
class ModelConfiguration:
    model_name = "google/electra-base-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    num_labels = 24
    electra_model = ElectraForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    first_hidden_dim = 256
    output_dim = num_labels


class SentimentClassifier(nn.Module):
    def __init__(self, electra_model, num_labels):
        super(SentimentClassifier, self).__init__()
        self.electra = electra_model
        self.drop = torch.nn.Dropout(0.4)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        pooled_output = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )[0]
        output = self.drop(pooled_output)
        return output


# if __name__ == "__main__":
#     modelConfig = ModelConfiguration()
#     tokenizer = modelConfig.tokenizer
#     electraModel = modelConfig.electra_model
#     model = SentimentClassifier(electraModel, modelConfig.num_labels)
#     print(model)

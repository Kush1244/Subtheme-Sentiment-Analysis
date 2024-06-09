import torch
import configuration as config
from transformers import BertModel, BertPreTrainedModel, BertTokenizer


# Bert Pretrained model with final classifier
class SentimentClassifierBERT(BertPreTrainedModel):
    def __init__(self, num_labels, conf):
        super(SentimentClassifierBERT, self).__init__(conf)
        self.bert = BertModel.from_pretrained(
            config.BERT_PRE_TRAINED_MODEL, return_dict=False
        )
        self.drop = torch.nn.Dropout(0.4)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        output = self.drop(pooled_output)
        output = self.classifier(output)
        return output

import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from transformers import BertTokenizer

model = "base"  # bert model type
INPUT_FILE = "input/Evaluation-dataset.csv"  # path of input file
MAX_LEN = 256  # Maximum length of Electra Tokenizer is 512
NUM_LABELS = 24  # Numbers of labels to predict
MODEL_PATH = "output/electra_model.pth.tar"  # save model path
ENCODER_PATH = "output/encoder.pkl"  # save label encoder path
LEARNING_RATE = 1e-04
EPOCHS = 5  # no. of epochs for training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "google/electra-base-discriminator"
TOKENIZER = ElectraTokenizer.from_pretrained(model_name)

electra_model = ElectraForSequenceClassification.from_pretrained(
    model_name, num_labels=NUM_LABELS
)

BERT_PRE_TRAINED_MODEL = "bert-base-uncased"
BERT_MODEL_PATH = "output/bert_model.pth.tar"
bert_tokenizer = BertTokenizer.from_pretrained(BERT_PRE_TRAINED_MODEL)
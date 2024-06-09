# Subtheme-Sentiment-Analysis
# Sentiment Analysis Project

This repository contains code for a sentiment analysis project using fine-tuned BERT and Electra models. The project includes scripts for data preprocessing, model fine-tuning, and inference.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt

Data Preprocessing
Before training the models, you need to preprocess the dataset. Start by running the preprocess_dataset.ipynb notebook. This notebook will generate pickle files for model training and validation.

Model Fine-Tuning
Once the dataset is preprocessed, you can fine-tune the models using the provided scripts. Run the following scripts:

fine_tune.py: Fine-tunes the Electra model.
fine_tune_bert.py: Fine-tunes the BERT model.
Inference
To use the fine-tuned models for inference, you can run the following scripts:

inference_electra.py --text "Replace with your text": Uses the fine-tuned Electra model for inference on the provided text.
inference_bert.py --text "Replace with your text": Uses the fine-tuned BERT model for inference on the provided text.
Replace "Replace with your text" with the text you want to analyze sentiment for.

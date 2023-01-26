import os
import re

from app.api import Input, Output
from transformers import pipeline,AutoTokenizer, BertModel
import torch

def predict_sentiment_request(input: Input) -> Output:
    # load model
    model = torch.load("model/bert.pt")
    tokenizer = torch.load("model/bert-tokenizer.pt")
    generator = pipeline('sentiment-analysis', model=model,tokenizer=tokenizer)

    output = generator(input.text)
    if output[0]["label"] == "neutral":
        output_label = "Neutral"
    else:
        output_label = output[0]["label"][:-1].capitalize()
    return Output(text = output_label)
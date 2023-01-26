import os
import re

from app.api import Input, Output
from transformers import pipeline
import torch

def predict_sentiment_request(input: Input) -> Output:
    # load model
    model = torch.load("model/bert.pt")
    tokenizer = torch.load("model/bert-tokenizer.pt")
    generator = pipeline('sentiment-analysis', model=model,tokenizer=tokenizer)

    output = generator(input.text)
    return Output(sentiment =  output[0]["label"])
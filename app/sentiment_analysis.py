import os
import re

from app.api import Input, Output
from transformers import pipeline

def sentiment_analysis():
    model_name = "oliverguhr/german-sentiment-bert"
    sentiment_analysis_model = pipeline(model=model_name, tokenizer=model_name)
    return sentiment_analysis_model

def predict_sentiment_request(input: Input) -> Output:
    model = sentiment_analysis()
    output = model(input.text)
    if output[0]["label"] == "neutral":
        output_label = "Neutral"
    else:
        output_label = output[0]["label"][:-1].capitalize()
    return Output(score = round(output[0]['score'], 4))
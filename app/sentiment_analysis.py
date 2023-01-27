import torch
from transformers import pipeline

from app.api import Input, Output


def get_generator():
    # load model
    model = torch.load("model/roberta.pt")
    tokenizer = torch.load("model/roberta-tokenizer.pt")
    return pipeline('sentiment-analysis', model=model,tokenizer=tokenizer)


def predict_sentiment_request(input: Input, generator) -> Output:
    #inputs = tokenizer(input.text, return_tensors="pt")
    #with torch.no_grad():
    #    logits = model(**inputs).logits
    #predicted_class_id = logits.argmax().item()
    output = generator(input.text)
    #return Output(sentiment =  model.config.id2label[predicted_class_id])
    return Output(sentiment = output)
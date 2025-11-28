from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pprint import pprint

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def sentiment_analyser(docs:list[str]) :

    encoded_docs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")

    model_inputs = torch.tensor(encoded_docs["input_ids"])
    model_output = model(model_inputs)

    softmax_output = torch.nn.functional.softmax(model_output.logits)

    sentiments = []
    for probs in softmax_output :
        index = 0 if probs[0] > probs[1] else 1
        sentiments.append(model.config.id2label[index])
        
    return sentiments

if __name__ == "__main__" :

    reviews = [
        "I really liked the individual performances but the overall movie was boring",
        "Items look and feel was decent. Overall its worth the money",
        "Farhan Qureshi has captured some stunning photos of Tigers in the wild"
    ]

    sentiments = sentiment_analyser(reviews)

    output_map = [ {'review': review, 'sentiment': sentiment} for review, sentiment in zip(reviews, sentiments)]
    print("--"*30)
    print("\nResults : \n")
    pprint(output_map)
        


from transformers import pipeline
from pprint import pprint

def sentiment_analysis(text_list:list[str], model:str = None) :
    sentiment_classifier = None
    if model :
        sentiment_classifier = pipeline(task='sentiment-analysis', model=model)
    else :
        sentiment_classifier = pipeline(task='sentiment-analysis')
    
    return sentiment_classifier(text_list)

def zero_shot_classifier(text_list:list[str], labels:list[str], model=None) :
    classifier = None
    if model :
        classifier = pipeline(task='zero-shot-classification', model=model)
    else :
        classifier = pipeline(task='zero-shot-classification')

    return classifier(text_list, candidate_labels=labels)

def translator(text:str, target_language:str, model:str = None) :
    translator = None
    if model :
        translator = pipeline(task='translation', model=model)
    else :
        translator = pipeline(task='translation')
    
    translator(text, tgt_lang=target_language)
    
    

if __name__ == "__main__" :

    reviews = [
        "I really liked the individual performances but the overall movie was boring",
        "Items look and feel was decent. Overall its worth the money",
        "Farhan Qureshi has captured some stunning photos of Tigers in the wild"
    ]

    sentiments = sentiment_analysis(reviews)
    print(f"\nSentiment Analysis : ")
    pprint(sentiments)

    classes = zero_shot_classifier(reviews, ["E-commerce","Entertainment","Photography"])
    print(f"\nZero shot classifier :")
    pprint(classes)

    translations = translator(reviews[1], "Hindi", "sarvamai/sarvam-translate")
    print(f"\nTranslator :")
    pprint(translations)


    
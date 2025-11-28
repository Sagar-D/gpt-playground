from transformers import AutoTokenizer
from pprint import pprint

TOKENISER_MODEL = "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(TOKENISER_MODEL)

def tokenize_single_step(docs:list[str]) :
    print("\n" + "--"*30)
    print("\nTokenization using tokinizer() __call__ method. (tokenize_single_step)")
    
    model_inputs = tokenizer(docs, padding=True, truncation=True)
    pprint(model_inputs)
    print("\n" + "--"*30)
    
    return model_inputs

def tokenize_multi_step(docs:list[str]) :
    
    print("\n" + "--"*30)
    print("\n Running Tokenization by individual steps (tokenize_multi_step)")

    print("\n Step 1 : Break sentences to tokens")
    tokens = [tokenizer.tokenize(doc) for doc in docs]
    print(tokens)

    print("\n Step 2 : Convert to token to token_ids")
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    print(token_ids)

    print("\n" + "--"*30)

    return token_ids

if __name__ == "__main__" :

    docs = [
        "Hello! Welcome to tokenization tutorials",
        "Lets tokenize!!"
    ]

    model_inputs = tokenize_single_step(docs)
    token_ids = tokenize_multi_step(docs)

    
    print("--"*30)
    
    print("Output from tokenize_single_step()\n")
    print(f"Token Ids : {model_inputs['input_ids']}\n")
    print(f"Token Ids : {[tokenizer.decode(token_id) for token_id in model_inputs['input_ids']]}\n\n")

    print("Output from tokenize_multi_step()\n")
    print(f"Token Ids : {token_ids}\n")
    print(f"Token Ids : {[tokenizer.decode(token_id) for token_id in token_ids]}\n")

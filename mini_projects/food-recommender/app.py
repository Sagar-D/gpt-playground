import json
from food_data_helper import get_formatted_food_data
from chatbot import chat_chain
from chroma_store import (
    delete_collection_if_exists,
    create_or_get_collection,
    add_documents
)

COLLECTION_NAME = "food_collection"

foods = ""
with open("mini_projects/food-recommender/data/FoodDataSet.json") as file:
    foods = json.load(file)

if type(foods) == "str":
    print("Failed to load JSON food data")
else:
    delete_collection_if_exists(COLLECTION_NAME)
    collection = create_or_get_collection(COLLECTION_NAME)
    add_documents(collection, get_formatted_food_data(foods))
    print(f"*** Info : {collection.count()} documents stored in Vector Store. ***")

print("\n\n" + ("--" * 30))
print("\n!!! Welcome to Food Recommender Chatbot !!!")
print("Ask your query related to food with me and I'll help you out")

while True:
    
    user_input = input("User : ")
    print("\n")

    if user_input.strip().lower() in ["bye", "quit", "exit"]:
        print("\nFood Denie : Bye! See you next time!\n")
        break

    response = chat_chain.invoke({"collection": collection, "prompt": user_input})
    print(f"\nFood Genie : {response}\n")

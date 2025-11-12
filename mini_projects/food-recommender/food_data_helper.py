from chromadb import Collection
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from prompt_templates import metadat_filter_extraction_prompt_template
import os
import dotenv

dotenv.load_dotenv()


def create_doc_string(food_obj):
    return f"""
Food Name : {food_obj['food_name']}
Food Description : {food_obj['food_description']}
Ingredients used for preperation of this dish : {" ".join([food+"," for food in food_obj['food_ingredients']])}

It is a {food_obj['cuisine_type']} cusine, cooked by {food_obj['cooking_method']}
Food's {", ".join([ key + " is " + food_obj['food_features'][key] for key in food_obj['food_features']])}

Health benifits of this dish : {food_obj['food_health_benefits']}
This food provides {food_obj['food_calories_per_serving']} calories per serving.
This food has nutritional value of {food_obj['food_nutritional_factors']['carbohydrates']} of carbohydrates, {food_obj['food_nutritional_factors']['protein']} of protein and {food_obj['food_nutritional_factors']['fat']} of fats.
"""


def remove_duplicates(foods):

    ids = set()
    filtered_foods = []

    for food in foods:
        if food["food_id"] not in ids:
            filtered_foods.append(food)
        ids.add(food["food_id"])

    return filtered_foods


def create_metadata(food_obj):
    return {
        "id": food_obj["food_id"],
        "name": str(food_obj["food_name"]).lower().strip(),
        "cooking_method": str(food_obj["cooking_method"]).lower().strip(),
        "cuisine_type": str(food_obj["cuisine_type"]).lower().strip(),
    }


def get_formatted_food_data(foods):

    filtered_foods = remove_duplicates(foods=foods)
    return [
        {
            "document": create_doc_string(food),
            "metadata": create_metadata(food),
            "id": "food_" + str(food["food_id"]),
        }
        for food in filtered_foods
    ]


def get_metatdat_filters(prompt: str, collection: Collection):

    llm = ChatOllama(
        base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0
    )

    metadata_filter_extraction_chain = (
        metadat_filter_extraction_prompt_template | llm | JsonOutputParser()
    )
    result_json = metadata_filter_extraction_chain.invoke({"prompt": prompt})
    print(f"Features extracted by llm : {result_json}")

    metadata_filter = {}
    cuisine_filter = {}
    cooking_method_filter = {}

    if "cuisine_type" in result_json and result_json["cuisine_type"] and str(result_json["cuisine_type"]).strip().lower() not in ["none","null"] :
        cuisine_filter["cuisine_type"] = (
            str(result_json["cuisine_type"]).lower().strip()
        )
        if len(collection.get(where=cuisine_filter)["documents"]) == 0:
            cuisine_filter = {}

    if "cooking_method" in result_json and result_json["cooking_method"] and str(result_json["cooking_method"]).lower().strip() not in ["none","null"]:
        cooking_method_filter["cooking_method"] = (
            str(result_json["cooking_method"]).lower().strip()
        )
        if len(collection.get(where=cooking_method_filter)["documents"]) == 0:
            cooking_method_filter = {}

    if len(cuisine_filter) == 0:
        if len(cooking_method_filter) == 0:
            return None
        else:
            return cooking_method_filter
    elif len(cooking_method_filter) == 0:
        return cuisine_filter

    metadata_filter["$and"] = []
    metadata_filter["$and"].append({"cuisine_type": cuisine_filter["cuisine_type"]})
    metadata_filter["$and"].append({"cooking_method": cooking_method_filter["cooking_method"]})

    if len(documents=collection.get(where=metadata_filter)["documents"]) == 0:
        if len(cuisine_filter) != 0:
            return cuisine_filter
        elif len(cooking_method_filter) != 0:
            return cooking_method_filter
        else:
            return None

    return metadata_filter

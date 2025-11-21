from langchain_core.prompts import ChatPromptTemplate

metadat_filter_extraction_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are data retriever helper. You help by extracting the metadata filters that can be applied for filtering based on user prompt.
Note : You only respond in JSON object. Don't add any additional information other than JSON object in the response.""",
        ),
        (
            "user",
            """Below are the metatdata filters available in our retriever system :
1. cuisine_type (String value representing the cuisine type. Supprted Values : American, Italian, French, Middle Eastern, Australian, British, International, Thai, German, Southern, Latin , Spanish, Korean, Greek, Mexican, Canadian, Universal, Japanese, Indian, Chinese.)
2. cooking_method (String value representing cooking method used. Supported Values : Baking, No-bake, Freezing, Steaming, Frying, Chilling, Mixing, Fermenting, Manufacturing, Boiling, Pickling, Extracting, Stewing.)

Your job is to go through the below given user prompt and extract values for above metadata filtes from the prompt, of available.
Once you extract the filter values, respond in the below JSON format. If user prompt doesn't have any specific value, ignore that field.


User Prompt : {prompt}

Instruction Format : Respond in the below JSON format
{{
    "cuisine_type": "str" (Value datatype : String. Supprted Values : American, Italian, French, Middle Eastern, Australian, British, International, Thai, German, Southern, Latin , Spanish, Korean, Greek, Mexican, Canadian, Universal, Japanese, Indian, Chinese. If value not found,  set it to JSON keyword null (without quotes)),
    "cooking_method" : "str" (Value datatype : String. Supported Values : Baking, No-bake, Freezing, Steaming, Frying, Chilling, Mixing, Fermenting, Manufacturing, Boiling, Pickling, Extracting, Stewing. If value not fount, set it to JSON keyword null (without quotes))
}}""",
        ),
    ]
)


youtube_qna_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are friendly chatbot who answers user queries related to youtube video transcript based on context provided. You are polite and always try to help the user only based on the provided context.

Important Note : If user asks anything that is not related to context shared or if not enough context is available for you to respond the user query, you can respond by saying - "I am not able to help you on this query due to lack of context. Is there anything else I can help you with".
**Do Not Hallucinate or generate any new information. Use only the information provided in the contex.**""",
        ),
        (
            "user",
            """Answer the user query based on the context provided.

Context : {context}


User Query : {prompt}""",
        ),
    ]
)

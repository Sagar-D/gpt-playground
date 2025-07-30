from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os

dotenv.load_dotenv()

llm = ChatOllama(base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0)

parser = StrOutputParser()

capital_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("human", "India"),
        ("ai", "DELHI maga"),
        ("human", "China"),
        ("ai", "BEIJING maga"),
        ("human", "USA"),
        ("ai", "WASHINGTON DC maga"),
        ("human", "Pakistan"),
        ("ai", "ISLAMABAD maga"),
        ("human", "Nepal"),
        ("ai", "KATHMANDU maga"),
        ("human", "Indonesia"),
        ("ai", "JAKARTA maga"),
        ("human", "{prompt}")
    ]
)

print("\n\n", "Few Shot Prompting using hardcoded examples".title(), "\n", sep="")
print("Enter a country name to know its capital : ", end="")
country = input()
chain = capital_prompt_template | llm | parser
print("\n", chain.invoke({"prompt": country}), sep="")

print("\n\n", "--" * 50, sep="")
print("\n\n", "Few Shot Prompting using examples from Data Set".title(), "\n", sep="")

data_set = [
    {"country": "India", "capital": "Delhi"},
    {"country": "China", "capital": "Beijing"},
    {"country": "USA", "capital": "Washington DC"},
    {"country": "Pakistan", "capital": "Islamabad"},
    {"country": "Indonesia", "capital": "Jakarta"}
]

data_set = list(
    map(lambda dict: {**dict, "text": f"Ayyy maga, {dict["capital"]} kano"}, data_set)
)

print("Updated Data set with custem text : ")
for data in data_set:
    print(data)

print("\n\n")


few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=data_set,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{country}"), ("ai", "{text}")]
    ),
)

capital_prompt_template = ChatPromptTemplate.from_messages(
    [few_shot_prompt, ("human", "{prompt}")]
)

print("Enter a country name to know its capital : ", end="")
country = input()
capital_predictor_chain = capital_prompt_template | llm | parser
print("\n", capital_predictor_chain.invoke({"prompt": country}), sep="")
print("\n\n")

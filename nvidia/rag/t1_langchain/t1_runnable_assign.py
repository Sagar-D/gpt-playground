from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnableAssign
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import dotenv
from pprint import pprint

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=1
)

llm_chain = llm | StrOutputParser()


def custom_print(x, prefix="Current State : "):
    print("--"*50)
    print(prefix)
    pprint(x)
    print("--"*50)
    print()
    return x


custom_print = RunnableLambda(custom_print)

classifier_propmt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Statement : {input_text}\n Classify the statement into one of the options : ['Boat', 'Car', 'Airplane']
Respond back with only one word""",
        )
    ]
)

sentence_generation_prompt = ChatPromptTemplate.from_messages(
    [("human", "Generate a single line sentence on the topic {topic}")]
)

paragraph_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Generate a paragraph on the topic {topic}."
            + " Start your paragraph with below sentence"
            + " and build on the same idea.\n Sentence : {sentence}",
        )
    ]
)

dict_based_chain = (
    custom_print
    | {
        "input_text": lambda x: x["input_text"],
        "topic": (classifier_propmt | llm_chain),
    }
    | custom_print
    | {
        "input": lambda x: x["input_text"],
        "topic": lambda x: x["topic"],
        "sentence": (sentence_generation_prompt | llm_chain),
    }
    | custom_print
    | paragraph_generation_prompt | llm_chain
)


print("--"*50, " "*10 + "DICT BASED CHAIN", "--"*50, sep="\n")

print("Final Response with Map Based Chain : \n", dict_based_chain.invoke({"input_text": "I like to fly high"}))



runnable_assign_chain = (
    custom_print
    | RunnableAssign({'topic': (classifier_propmt | llm_chain)})
    | custom_print
    | RunnableAssign({'sentence': (sentence_generation_prompt | llm_chain)})
    | custom_print
    | (paragraph_generation_prompt | llm_chain)
)

print("--"*50, " "*10 + "RUNNABLE ASSIGN BASED CHAIN", "--"*50, sep="\n")

print("Final Response with Runnable Assign Chain : \n", runnable_assign_chain.invoke({"input_text": "I got sea sick after 3 days of commute"}))



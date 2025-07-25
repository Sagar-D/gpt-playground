from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import dotenv
import os

dotenv.load_dotenv()


class LangChain:

    def __init__(
        self,
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL"),
        temperature=0,
    ):
        self.model = model
        self.base_url = base_url
        self.llm = ChatOllama(base_url=base_url, model=model, temperature=temperature)


if __name__ == "__main__":

    # create a llm runnable
    client = LangChain()

    # create a prompt template runnable
    translate_template = ChatPromptTemplate.from_template(
        """\
Translate the below text from {source_language} to {target_language}.\
Please respond with only translation, don't add any other notes or comments.
{text}
"""
    )

    # create a string parser runnable
    parser = StrOutputParser()

    # create a chain of all the runnables
    chain_with_parser = translate_template | client.llm | parser

    print("\nBelow is the graph of the chain with Parser")
    print(chain_with_parser.get_graph().draw_ascii())
    print("\nBelow is the schema showing parameters and reqauired params : \n")
    print(json.dumps(chain_with_parser.input_schema.model_json_schema(), indent=4))

    print("--" * 50)
    print("\n\n\n")

    print("Welcome to the Language Translator!!")

    print("Enter the source language : ", end="")
    source_language = input()

    print("Enter the target language : ", end="")
    target_language = input()

    print(f"Enter the text in {source_language} for translation : ", end="")
    text = input()

    print("Below is the translated text : ")

    # Note that response from llm is auto parsed to string by String parser
    # runnable added at the last leg of the chain
    print(
        chain_with_parser.invoke(
            {
                "source_language": source_language,
                "target_language": target_language,
                "text": text,
            }
        )
    )

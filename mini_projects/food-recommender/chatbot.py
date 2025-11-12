from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser

from prompt_templates import food_recommender_prompt_template
from chroma_store import query_docs

import os
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0.5
)

# Chains
chat_chain = (
    RunnableMap(
        {
            "prompt": lambda data: data["prompt"],
            "context": lambda data: "\n\n".join(
                query_docs(data["collection"], data["prompt"])
            ),
        }
    )
    | food_recommender_prompt_template
    | llm
    | StrOutputParser()
)

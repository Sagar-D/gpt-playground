from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from prompt_templates import youtube_qna_prompt_template
from retriever import retrieve_docs

import os
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
    temperature=0.5
)

chat_chain = (
    RunnableMap(
        {
            "prompt" : lambda data : data["prompt"],
            "context" : lambda data : "\n\n".join([doc.text for doc in retrieve_docs(data["vector_index"], data["prompt"])][:2])
        }
    ) | youtube_qna_prompt_template | llm | StrOutputParser()
)

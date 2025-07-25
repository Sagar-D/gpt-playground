from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
    temperature=0
)

parser = StrOutputParser()


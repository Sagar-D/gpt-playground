from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableAssign
from langchain_community.document_loaders import TextLoader
import os
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=1
)

documents = TextLoader("nvidia/rag/document_rag/sample_docs/rahul_dravid.txt").load()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant. Your job is to answer the user query based on the context provided to you",
        ),
        ("human", "User query : {user_prompt}\nContext : {context}"),
    ]
)

chain = (
    RunnableAssign({"context": lambda x: documents[0].page_content})
    | prompt
    | llm
    | StrOutputParser()
)

query = "What is Rahul Dravid known as?"
print(f"\n\n[Human] : {query}")
print(f"\n[Assistant] : {chain.invoke({"user_prompt": query})}")

query = "What are Rahul Dravid's current plan"
print(f"\n\n[Human] : {query}")
print(f"\n[Assistant] : {chain.invoke({"user_prompt": query})}")

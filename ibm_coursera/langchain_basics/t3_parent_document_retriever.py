from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
from langchain.storage import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate

import re
import os
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0.5
)

web_page = "https://www.livescience.com/11375-top-ten-conspiracy-theories.html"
web_doc_loader = WebBaseLoader(web_page)
documents = web_doc_loader.load()

for document in documents:
    document.page_content = re.sub("\n+", "\n", document.page_content)
    document.page_content = re.sub("[ \t]+", " ", document.page_content)

parent_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200
)
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db_path = "vector_db/parent_doc_retrieval"
vector_db_collection_name = "pdc_child_collection"

## Clear older data in Chrome Vector store
shutil.rmtree(vector_db_path, ignore_errors=True)

vector_store = Chroma(
    embedding_function=embedder,
    collection_name=vector_db_collection_name,
    persist_directory=vector_db_path,
)
parent_doc_store = InMemoryStore()

parent_doc_retriever = ParentDocumentRetriever(
    child_splitter=child_text_splitter,
    parent_splitter=parent_text_splitter,
    vectorstore=vector_store,
    docstore=parent_doc_store,
)

parent_doc_retriever.add_documents(documents=documents)

query = "Princess Diana"
searched_doc = vector_store.similarity_search(query=query, k=1)
retrieved_doc = parent_doc_retriever.invoke(query)

print("Result from Similarity Search on Vector Store")
print(f"Docs fetched : {len(searched_doc)}")
print("--" * 50)
print(searched_doc[0].page_content)

print("Result from Parent Document Retrieval")
print(f"Docs fetched : {len(retrieved_doc)}")

for i, doc in enumerate(retrieved_doc) :
    print("--" * 50)
    print(f"Dcoument {i+1}")
    print("--" * 50)
    print(doc.page_content)

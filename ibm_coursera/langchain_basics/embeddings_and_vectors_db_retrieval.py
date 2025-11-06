from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import dotenv
import os
import re

## Load .env
dotenv.load_dotenv()

## Create LLM instance
llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0.5
)

## Load Web page to documents
web_page = "https://www.livescience.com/11375-top-ten-conspiracy-theories.html"
vector_collection_name = "international_conspiracies"
web_document_loader = WebBaseLoader(web_page)
document = web_document_loader.load()
print("Downloaded Web page content...")

## Remove extra spaces and newlines
for page in document:
    page.page_content = re.sub(r"[\n]+", "\n", page.page_content)
    page.page_content = re.sub(r"[ \t]+", " ", page.page_content)

## Document chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = text_splitter.split_documents(documents=document)
print("Document chunking completed...")

## Create Embeddings of document chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
print("Embeddigns created...")

## Clear older data in Chrome Vector store
shutil.rmtree("conspiracy_db", ignore_errors=True)

## Create unique document ids
doc_ids = [f"doc_{i}" for i in range(len(embedding_vectors))]

## Store documents and embeddings in Chroma Vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    ids=doc_ids,
    persist_directory="conspiracy_db",
    collection_name=vector_collection_name,
)
vector_store.persist()
print("Embeddingas nad documents stored in Chroma Vector DB...")

## Query Chroma vector store for similarity search
query_string = "Who claimed that Princess Diana's assasination was plotted by British Intelligence?"

## Manually retrieve similar documents by similarity search
# result_documents = vector_store.similarity_search(query=query_string, k=3) # k=3 => Fetch top 3 chunks/documents
# for doc in result_documents:
#     print("--" * 50)
#     print(doc.page_content + "\n")

### Create a retriever instance
retriever = vector_store.as_retriever(search_kwargs={'k':3})

### Create a prompt template for query
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly chatbot who provides answers to the user queries based on the context provided to you.
If the passed on context is not sufficient to answer the query, you should prompt;y respond to the user saying you don't have the context to answer the query"""),
    ("user", "Answer the query using below context.\n\nContext : {context}\n\nQuery : {query}")
])

rag_chain = (
    RunnableMap(
        {
            "context" : (lambda x: x["query"]) | retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])),
            "query" : (lambda x: x["query"])
        }
    ) | prompt_template | llm | StrOutputParser()
)


print("Initiating LLM...")
print(f"\nAsk me anything related to this article and I'll answer you!!\n Article Link : {web_page}\nSend prompt 'Bye' to quit\n\n")

while True :
    query_string = input("Query : ")

    if query_string.lower() == "bye" :
        break
    llm_response = rag_chain.invoke({"query": query_string})
    print(f"Answer : {llm_response}\n\n")

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os
import re

dotenv.load_dotenv()

llm = ChatOllama(base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0.5)

web_page = "https://www.longlongtimeago.com/once-upon-a-time/folktales/punyakoti-the-cow-and-arbhuta-the-tiger"
web_loader = WebBaseLoader(web_page)
document = web_loader.load()

for page in document :
    page.page_content = re.sub(r"\n+", "\n", page.page_content)
    page.page_content = re.sub(r"[ \t]+", " ", page.page_content)


text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(document)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a an AI chatbot, who understands the stories provided to you in context and answer user  queries based on it."),
    ("user", "Here is the story - {story}"),
    ("user", "{prompt_text}")
])
prompt = prompt.partial(story= chunks[2].page_content)

chain = prompt | llm | StrOutputParser()

prompt_text = input("Ask your question related to Folktale of Punyakoti : ")

response = chain.invoke({"prompt_text": prompt_text})
print(f"\nAI Response : {response}")

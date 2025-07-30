from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser
import os
import dotenv

dotenv.load_dotenv()


llm = ChatOllama(base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0)

json_parser = SimpleJsonOutputParser()

book_details_prompt = ChatPromptTemplate.from_template(
    """\
Provide me a JSON data with book_title, author and year_of_publication for the book '{book_title}'.
Send only JSON data in the response. Don't add any non-JSON text, notes/details or even code blocks in the response.
"""
)

book_details_chain = book_details_prompt | llm | json_parser

books_list = [
    {"book_title": "Dune"},
    {"book_title": "Neuromancer"},
    {"book_title": "Snow Crash"},
    {"book_title": "The Left Hand of Darkness"},
    {"book_title": "Foundation"}
]

for book_details_json in book_details_chain.batch(books_list) :
    print(book_details_json)
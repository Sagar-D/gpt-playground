from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0
)

parser = StrOutputParser()

thesis_statements = [
    "The fundametal concepts quantum physcis are difficult to graps, even for the mostly advanced students.",
    "Einstein's theroy of relativity revolutionised undrstanding of space and time, making it clear that they are interconnected.",
    "The first law of thermodynmics states that energy cannot be created or destoryed, excepting only transformed from one form to another.",
    "Electromagnetism is one of they four funadmental forces of nature, and it describes the interaction between charged particles.",
    "In the study of mechanic, Newton's laws of motion provide a comprehensive framework for understading the movement of objects under various forces.",
]

## Fix the grammer and spelling errors for above thesis statements.
## Generate a paragraph for each of the thesis statements

spell_correct_template = ChatPromptTemplate.from_template(
    """\
Correct any spelling or grammatical errors in the below statement and respond back with the corrected statemnt. \
Respond on;y with the corrected statment and don'e send any other notes or details.
Statement : "{statment}"
"""
)

spell_correct_chain = spell_correct_template | llm | parser

paragraph_generator_template = ChatPromptTemplate.from_template(
    """\
Generate a paragraph related to the statement given below. Use the below statment as the first line of the paragraph. \
Respond only with paragraph. Don't add any other notes or details in the response.
Statemnet : "{statemnet}"
"""
)

paragraph_generator_chain = paragraph_generator_template | llm | parser

final_chain = spell_correct_chain | paragraph_generator_chain

for paragraph in final_chain.batch(thesis_statements):
    print(paragraph)
    print()

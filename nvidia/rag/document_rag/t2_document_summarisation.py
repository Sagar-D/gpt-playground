# Step 1 : Load the documents
# Step 2 : Refine the documents (create smaller chunks)
# Optional Step 3 : Pre-Process chunks
# Step 3 : Summarize all chunks to create a DocumentKnowledgeBase

import os
import dotenv
import ast
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableAssign
from typing import List
from functools import partial
from rich.console import Console
from rich.style import Style

def pprint(text, prefix="") :
    base_style = Style(color="#76B900", bold=True)
    Console().print(text, style=base_style)

def preparse_json(json_string):
    if '{' not in json_string: json_string = '{' + json_string
    if '}' not in json_string: json_string = json_string + '}'
    json_string = (json_string
        .replace("\\_", "_")
        .replace("\n", " ")
        .replace("\]", "]")
        .replace("\[", "[")
    )
    try:
        data = ast.literal_eval(json_string)
        return json.dumps(data)  # Clean, valid JSON string
    except Exception as e:
        print("--"*30)
        print("Error:", e)
        print("--"*30)
        return json_string

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0.1
)

# Step 1 : Load Document
pdf_doc = PyPDFLoader(
    "nvidia/rag/document_rag/sample_docs/india.pdf", extract_images=True
).load()

# Spep 2 : Create chunks of document
text_splitter = RecursiveCharacterTextSplitter(
    separators=[". ", "\n", "\n\n", " "], chunk_size=1000, chunk_overlap=50
)

pdf_doc_chunks = []

for page in pdf_doc:
    text_chunks = text_splitter.split_text(page.page_content)
    pdf_doc_chunks.append(text_chunks)

# print(len(pdf_doc_chunks))

# Step 3 : Summarize all chunks to create a DocumentKnowledgeBase


class DocumentKnowledgeBase(BaseModel):
    """Knowledge base of a document. Store the running summary of Document"""

    important_topics_in_current_chunk: List[str] = Field([], description="All the important points sound in the current chunk of information. Override the old value. (Maximum 3 items in the list)")
    running_summary: str = Field(
        "",
        description="Running summary of document. Information dense and captures data points. Will be used as document knowledge base. Always only append!, Never override! ",
    )

knowledge_base_field_descriptions = {
    name: field.description
    for name, field in DocumentKnowledgeBase.model_fields.items()
    if field.description is not None
}


knowledge_prompt = ChatPromptTemplate.from_messages([
    ("system", """\
You are a document summarisation engine, trying to go through the chunks of dcoument and create a knowledge base.
Keep appending the new knowledge you see to the running_summary in knowledge base. Do not override old knowledge, unless neccessary.
Keep the summary short and crisp, but at the same time it should be dense and informative.
Avoid keeping duplicate or redundant information.

Strictly return a JSON object with the following format:\n{format_instructions}\n
Do not add any other notes, comments or details in the reponse. ** Send only JSON object as response! **
"""),
    ("human", """\
Here is the curresnt state of knowledge base : {knowledge_base}
Go throught the below chunk of information from the document and update the knowledge base accordingly.
Note: Do not override the running_summary. Only update it by appending.
Capture the important datapoinst in a consice way in the running_summary. Keep in mind that, this running_summary should help you as base knowledge base for any queries asked by user in the future.
Information Chunk : {doc_chunk}""")
]
)

knowledge_output_parser = JsonOutputParser(pydantic_object=DocumentKnowledgeBase)

latest_knowledge = ""
state = {
    "knowledge_base" : latest_knowledge,
    "doc_chunk" : ""
}

knowledge_chain = (
    RunnableAssign({"knowledge_base": lambda state: latest_knowledge, "format_instructions": lambda state : knowledge_output_parser.get_format_instructions()})
    | RunnableAssign({"knowledge_base": (knowledge_prompt| llm | StrOutputParser() | preparse_json | knowledge_output_parser)})
)


for chunk in pdf_doc_chunks :
    pprint(f"\n{'--'*30}\n")
    pprint(state['knowledge_base'])
    print()
    state['doc_chunk'] = chunk
    state = knowledge_chain.invoke(state)
    latest_knowledge = state['knowledge_base']

pprint(f"\n{'--'*30}\n")
pprint(f"{' '*20}FINAL KNOWLEDGE BASE")
pprint(f"\n{'--'*30}\n")
pprint(latest_knowledge)
print()

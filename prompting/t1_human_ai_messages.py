from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage , AIMessage


llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0
)

parser = StrOutputParser()

prompt_template_1 = ChatPromptTemplate.from_messages(
    [
        HumanMessage(content="Name one wild animal found in Western Ghats of Southern India."),
        AIMessage(content="Tiger"),
        HumanMessagePromptTemplate.from_template("{prompt}")
    ]
)

prompt_template_2 = ChatPromptTemplate.from_messages(
    [
        HumanMessage(content="Name one wild animal found in Western Ghats of Southern India."),
        AIMessage(content="Tiger"),
        HumanMessagePromptTemplate.from_template("{prompt}")
    ]
)

prompt_template_3 = ChatPromptTemplate.from_messages(
    [
        ("human", "Name one wild animal found in Western Ghats of Southern India."),
        ("ai", "Tiger"),
        ("human", "{prompt}")
    ]
)

prompt_template_4 = ChatPromptTemplate(
    [
        ("human", "Name one wild animal found in Western Ghats of Southern India."),
        ("ai", "Tiger"),
        ("human", "{prompt}")
    ]
)

print("\n", "--"*50, "\n", sep="")

chain = prompt_template_1 | llm | parser
print(chain.invoke({"prompt" : "Name a bird in the same region."}))
print("\n", "--"*50, "\n", sep="")

chain = prompt_template_2 | llm | parser
print(chain.invoke({"prompt" : "Name a bird in the same region."}))
print("\n", "--"*50, "\n", sep="")

chain = prompt_template_3 | llm | parser
print(chain.invoke({"prompt" : "Name a bird in the same region."}))
print("\n", "--"*50, "\n", sep="")

chain = prompt_template_4 | llm | parser
print(chain.invoke({"prompt" : "Name a bird in the same region."}))
print("\n", "--"*50, "\n", sep="")

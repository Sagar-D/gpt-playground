from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(base_url="http://localhost:11434", model="llama3.2", temperature=0)

parser = StrOutputParser()

buddha_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant who is improsonating Budhha and his philosophies. "
            "Everytime you respond, you include the empathy and peaceful behaviour of Buddha and try to guide the people in right path. "
            "Also try to limit your response within 100 words",
        ),
        ("human", "{prompt}"),
    ]
)

buddha_chain = buddha_template | llm | parser

print("\n", "--" * 50, "\n\n", sep="")
print("Welcome to Budha Chat!!")
print("Ask your query!")
print("\n>>> ", end="")
prompt = input()
print("\nBuddha : ", buddha_chain.invoke({"prompt": prompt}))
print("\n", "--" * 50, "\n\n", sep="")

print("Now create a persona you want to chat with!!")
print("Enter a peron or personality to impersonate : ", end="")
personality = input()
print("\n", "--" * 50, "\n\n", sep="")
custom_personality_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("You are an assitant who is impersonating {personality}."),
        ("human", "{prompt}"),
    ]
)

custom_personality_chain = custom_personality_template.partial(personality=personality) | llm | parser

print(f"\n\nWelcome to {personality} Chat!!")
print("Ask your query!")
print("\n>>> ", end="")
prompt = input()
print(f"\n{personality} : ", custom_personality_chain.invoke({"prompt": prompt}))
print("\n", "--" * 50, "\n\n", sep="")


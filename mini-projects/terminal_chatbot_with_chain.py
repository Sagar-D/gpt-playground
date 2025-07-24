from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import json

llm = ChatOllama(base_url="http://localhost:11434", model="llama3.2", temperature=0)

parser = StrOutputParser()

chat_history = []


def user_prompt_manager(user_input):
    chat_history.append({"role": "user", "content": user_input})
    return chat_history


def chat_history_manager(llm_response):
    chat_history.append({"role": "assistant", "content": user_input})
    return llm_response


user_prompt_manager_runnable = RunnableLambda(user_prompt_manager)
chat_history_manager_runnable = RunnableLambda(chat_history_manager)

chat_bot_chain = (
    user_prompt_manager_runnable | llm | parser | chat_history_manager_runnable
)

if __name__ == "__main__":

    print("\n\nWelcome to your terminal chatbot!!")
    print("Note : To exit the chat, pass the prompt '/bye'")
    print("--" * 50)

    while True:
        print("\n>>>", end=" ")
        user_input = input()
        if user_input == "/bye":
            break
        response = chat_bot_chain.invoke(user_input)
        print(response)

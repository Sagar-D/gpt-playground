from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os

dotenv.load_dotenv()

llm = ChatOllama(base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0)

parser = StrOutputParser()

chat_history = []


def user_prompt_manager(user_input):
    chat_history.append({"role": "user", "content": user_input})
    return chat_history


def chat_history_manager(llm_response):
    chat_history.append({"role": "assistant", "content": llm_response})
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
        for chunk in chat_bot_chain.stream(user_input) :
            print(chunk)

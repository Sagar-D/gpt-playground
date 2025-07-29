from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os

dotenv.load_dotenv()


class ChatBot:

    def __init__(self, llm_base_url=os.getenv("LLM_BASE_URL"), llm_model=os.getenv("LLM_MODEL")):
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm = ChatOllama(
            base_url=llm_base_url,
            model=llm_model,
            temperature=0.9
        )
        self.chat_history = []

        self.chat_prmpt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a chatbot assistant. Try to understand the user's need and assist them with required information."),
                ("placeholder", "{chat_history}")
            ]
        )

        self.chat_chain = self.chat_prmpt_template | self.llm | StrOutputParser()
    
    def chat(self, prompt) :
        self.save_history("human", prompt)
        llm_response = ""
        for llm_response_chunk in self.chat_chain.stream({"chat_history": self.chat_history}) :
            yield llm_response_chunk
            llm_response += llm_response_chunk
        self.save_history("ai", llm_response)
    
    def save_history(self, message_by , message) :
        if len(self.chat_history) >= 10 :
            self.chat_history.pop(0)
        self.chat_history.append((message_by, message))

    def clear_history(self) :
        self.chat_history.clear()

if __name__ == "__main__":

    print("\n\nWelcome to your terminal chatbot!!")
    print("Note : To exit the chat, pass the prompt '/bye'")
    print("--" * 50)

    chatbot = ChatBot()

    while True:
        print("\n>>>", end=" ")
        user_input = input()
        if user_input.lower() == "/bye":
            break
        for chunk in chatbot.chat(user_input) :
            print(chunk, end="")
        print()

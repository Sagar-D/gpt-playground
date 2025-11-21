from langchain_ollama import ChatOllama
import os
import dotenv
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()


class Chatbot:

    def __init__(self, retriever):

        self.llm = ChatOllama(
            base_url=os.getenv("LLM_BASE_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=0.5,
        )
        self.retriever = retriever

    def chat(self, prompt):

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful chatbot who responds to user queries based on the context provided.\n"
                    "You answer only if you find relevent content for user query in the context provided. If you don't find right content in the context provide you will respond saying 'I don't have answer for your query"
                    "\n\nNote: While responding, Do not mention that you are refrencing the context for answering",
                ),
                ("user", "Query : {prompt}\n\n Context : {context}"),
            ]
        )

        chain = (
            RunnableMap(
                {
                    "prompt": (lambda x: x["input"]),
                    "context": (lambda x: x["input"])
                    | self.retriever
                    | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
                }
            )
            | prompt_template
            | self.llm
        )

        return chain.invoke({"input": prompt}).content

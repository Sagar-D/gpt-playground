from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from prompt_manager import MEETING_MOM_PROMPT
from speech_to_text import speech_to_text
import os
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    model=os.getenv("LLM_MODEL"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.5
)

mom_chain = RunnableLambda(speech_to_text) | MEETING_MOM_PROMPT | llm | StrOutputParser()

if __name__ == "__main__" :

    file_path = "mini_projects/meeting_mom_bot/data/sample_meeting_rec_1_min.wav"
    response = mom_chain.invoke(input=file_path)
    print(response)
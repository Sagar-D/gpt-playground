from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompt_manager import MASTER_SYSTEM_PROMPT, FETCH_NUTRITION_VALUE_PROMPT

import os
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    model=os.getenv("VISION_MODEL"),
    base_url= os.getenv("LLM_BASE_URL")
)

nutrition_analysis_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", MASTER_SYSTEM_PROMPT),
        ("human", [
            {"type" : "text", "text": FETCH_NUTRITION_VALUE_PROMPT},
            {"type": "image_url", "image_url": "{image_url}"}
        ])
    ]
)

nutritionist_chain = nutrition_analysis_prompt_template | llm | StrOutputParser()

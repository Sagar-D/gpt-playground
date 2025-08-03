from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableAssign
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, List, Union
from pprint import pprint
import os
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0
)


def print_knowledge_base(x, prefix="KNOWLEDGE_BASE STATE : "):
    print("--" * 50)
    print(" " * 20, prefix)
    pprint(x["knowlede_base"])
    print("--" * 50)
    print()
    return x


print_knowledge_base = RunnableLambda(print_knowledge_base)


class KnowledgeBase(BaseModel):
    """Store Knowledge base on user conversations"""

    # session_id: str = Field(description="Session Id of the User conversation")
    topic: str = Field(description="Main Topic of discussion")
    preferences: Dict[str, Union[str, int, float]] = Field(
        description="User preferences with key as subject/topic and value as preference value"
    )
    sentiment: str = Field(
        description="User Sentiment across chat in Positive or Negative"
    )


knowledge_base_field_descriptions = {
    name: field.field_info.description
    for name, field in KnowledgeBase.__fields__.items()
    if field.field_info.description is not None
}

knowledge_update_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI that responds ONLY with JSON following the given schema.""",
        ),
        (
            "human",
            """Given the below conversation history,
\n\nConversation History : 
{chat_history_string}
\n\n
Return a JSON object with the following format:\n{format_instructions}""",
        ),
    ]
)

FORMATTING_INSTRUCTIONS = (
    """The output should be formatted as a JSON instance that conforms to the JSON schema below.
Here is the output schema I am expecting for you to follow while generating response:
```SAMPLE_JSON_SCHEMA```
Note: Strictly follow above schema and do not add any other aditional data.
"""
).replace("SAMPLE_JSON_SCHEMA", str(knowledge_base_field_descriptions))

knowledge_update_prompt = knowledge_update_prompt.partial(
    format_instructions=FORMATTING_INSTRUCTIONS
)
knowledge_update_chain = (
    knowledge_update_prompt | llm | JsonOutputParser(pydantic_object=KnowledgeBase)
)


chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an conversational assistant AI."),
        ("placeholder", "{chat_history}"),
    ]
)

# Maintain Chat History
chat_history = []


def update_chat_history(data_obj):
    if data_obj["current_chat_actor"] == "human":
        chat_history.append(("human", data_obj["user_prompt"]))
    else:
        chat_history.append(("ai", data_obj["ai_response"]))
    return chat_history


update_chat_history = RunnableLambda(update_chat_history)

chain = (
    RunnableAssign({"current_chat_actor": lambda x: "human"})
    | RunnableAssign({"chat_history": update_chat_history})
    | RunnableAssign({"chat_history_string": lambda x: str(x["chat_history"])})
    | RunnableAssign({"knowlede_base": knowledge_update_chain})
    | print_knowledge_base
    | RunnableAssign(
        {
            "ai_response": (chat_prompt | llm | StrOutputParser()),
            "current_chat_actor": lambda x: "ai",
        }
    )
    | RunnableAssign({"chat_history": update_chat_history})
    | RunnableLambda(lambda x: x["ai_response"])
)

if __name__ == "__main__":

    while True:
        print(">>> ", end="")
        input_prompt = input()
        if input_prompt == "bye":
            break
        print(chain.invoke({"user_prompt": input_prompt}))

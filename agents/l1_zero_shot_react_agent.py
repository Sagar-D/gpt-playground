from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import AgentType, initialize_agent
import os
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL")
)

@tool
def add_numbers(inputs:str) -> str :
    """
    Calculate the sum of all the numbers passed comma separated integer string.
    
    Parameters :
    - inputs (str) : A string with comma separated numbers
    
    Return :
    - (str) : A string with result value after adding all the numbers

    Example input : "1, 2, 6"
    Example output : "9"
    """
    
    result = 0
    inputs = inputs.strip().strip('"').split(",")
    for num in inputs :
        if num.strip().isdigit() :
            result += int(num)
    
    return str(result)


tools = [add_numbers]
agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

SYSTEM_PROMPT = "\n\nNote : Do not get stuck in a loop of tool calling. Call the tool only once."
user_prompt = input("User Query : ")
response = agent.invoke(user_prompt + SYSTEM_PROMPT)

print("--"*30)
print(f"\n\nAgent Response : {response['output']}")
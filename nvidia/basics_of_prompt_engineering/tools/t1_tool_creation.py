from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
import os
import dotenv
from pprint import pprint

dotenv.load_dotenv()

llm = ChatOllama(base_url=os.getenv('LLM_BASE_URL'), model=os.getenv('LLM_MODEL'), temperature=0)

# Schema for Add Tool
class AddToolSchema(BaseModel) :
    """Use this **only** if you need to add two numbers. Not for general questions."""
    num1: int = Field(..., description="first number")
    num2: int = Field(..., description="second number")

# Addion tool - Adds 2 integers
@tool(args_schema=AddToolSchema)
def add(num1, num2) :
    return num1 + num2

# Schema for multiply tool
class MultiplyToolSchema(BaseModel) :
    """Use this **only** if you need to multiply two numbers. Not for general questions."""
    num1: int = Field(..., description="first number")
    num2: int = Field(..., description="second number")

# Multiplication Tool - Multiply 2 integers
@tool(args_schema=MultiplyToolSchema)
def multiply(num1, num2) :
    return num1 * num2
    

# Bind both the tools to llm
llm_with_tool_binding = llm.bind_tools([add, multiply])

system_message_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant capable of tool calling when helpful, necessary, and appropriate.

Think hard about whether or not you need to call a tool, \
based on your tools' descriptions and use them, but only when appropriate!

Whether or not you need to call a tool, address the user's query in a helpful informative way."""),
    ("human", "{prompt}")
])



def tool_caller(llm_response) :
    if len(llm_response.tool_calls) == 0 :
        return llm_response.content
    
    tools_map = {
        'add': add,
        'multiply': multiply
    }

    selected_tool = tools_map[llm_response.tool_calls[0]['name']]
    return selected_tool.invoke(llm_response.tool_calls[0]['args'])

chain = system_message_prompt | llm_with_tool_binding | RunnableLambda(tool_caller)


llm_response = chain.invoke({'prompt': "What is the sum of 12445 and 14325"})
print(f"Sum : {llm_response}")

llm_response = chain.invoke({'prompt': "What is the product of 12445 and 14325"})
print(f"Product : {llm_response}")

# Note : Tool is hallucinating and using tools for every query even after add system message
# Need to understand why and how to fix it
llm_response = chain.invoke({'prompt': "Why earth is not flat"})
print(f"Random Question : {llm_response}")

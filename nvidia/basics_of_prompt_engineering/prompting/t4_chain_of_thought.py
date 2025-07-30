from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0
)

parser = StrOutputParser()


### Chain of Thpught Prompting

example_problem = "What is 678 * 789?"
example_chain_of_thought = """\
Let me break this down into steps. First I'll break down 789 into hundreds, tens, and ones:

789 -> 700 + 80 + 9

Next I'll multiply 678 by each of these values, storing the intermediate results:

678 * 700 -> 678 * 7 * 100 -> 4746 * 100 -> 474600

My first intermediate result is 474600.

678 * 80 -> 678 * 8 * 10 -> 5424 * 10 -> 54240

My second intermediate result is 54240.

678 * 9 -> 6102

My third intermediate result is 6102.

My three intermediate results are 474600, 54240, and 6102.

Adding the first two intermediate results I get 474600 + 54240 -> 528840.

Adding 528840 to the last intermediate result I get 528840 + 6102 -> 534942

The final result is 534942.
"""

chain_of_thought_prompt_template = ChatPromptTemplate.from_messages(
    {
        ("human", example_problem),
        ("ai", example_chain_of_thought),
        ("human", "{prompt}"),
    }
)

cot_chain = chain_of_thought_prompt_template | llm | parser

print("\n\nChain Of Thought Prompting...\n")
print("Provide a 3 digit multiplication problem to llm : ", end="")
prompt = input()
print(cot_chain.invoke({"prompt": prompt}))

### Zero Shot Chain of Thpught Prompting

zero_shot_cot_prompt_template = ChatPromptTemplate.from_template(
    """{prompt}
Let's think step by step."""
)

zero_shot_cot_chain = zero_shot_cot_prompt_template | llm | parser

print("\n\nZero-Shot Chain Of Thought Prompting...\n")
print("Provide a 3 digit multiplication problem to llm : ", end="")
prompt = input()
print(zero_shot_cot_chain.invoke({"prompt": prompt}))


### Assignment

word_problem = """Michael's car travels at 40 miles per hour. He is driving from 1 PM to 4 PM and then \
travels back at a rate of 25 miles per hour due to heavy traffic. How long in \
terms of minutes did it take him to get back?"""
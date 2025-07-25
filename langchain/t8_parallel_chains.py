from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0
)

parser = StrOutputParser()

statements = [
    "I had a fantastic time hiking up the mountain yesterday.",
    "The new restaurant downtown serves delicious vegetarian dishes.",
    "I am feeling quite stressed about the upcoming project deadline.",
    "Watching the sunset at the beach was a calming experience.",
    "I recently started reading a fascinating book about space exploration.",
]

## For the above statements, perform below actions
## 1. Analyse Overall sentiment
## 2. Find the Main topic of the statment
## 3. Generate a Followup question
## Format the above information along with statement and print the formatted response
## Use parallel chains where applicable

sentiment_template = ChatPromptTemplate.from_template(
    """\
Analyse the sentiment of the below statemnet and respond back in one word as either "Positive" or "Negative".
Statement : {statment}
"""
)

main_topic_template = ChatPromptTemplate.from_template(
    """\
Find the main topic under discussion from the below statment and respond back with only the main topic. \
Don't add any other notes or details in the response.
Statement : {statment}
"""
)

follow_up_template = ChatPromptTemplate.from_template(
    """\
Ask a follow up question for the below statemnet to understand the missing information in the statment. \
Respond with only question. Don't add any other notes or details in the response.
Statement : {statment}
"""
)

sentiment_chain = sentiment_template | llm | parser
main_topic_chain = main_topic_template | llm | parser
follow_up_chain = follow_up_template | llm | parser

parallel_chain = RunnableParallel(
    {
        "sentiment": sentiment_chain,
        "main_topic": main_topic_chain,
        "follow_up": follow_up_chain,
        "statement": RunnableLambda(lambda statemnt: statemnt.capitalize()),
    }
)

custom_output_parser = RunnableLambda(
    lambda response: (
        f"Statement : {response['statement']}\n"
        f"Sentiment : {response['sentiment']}\n"
        f"Main Topic : {response['main_topic']}\n"
        f"Follow Up Question : {response['follow_up']}\n"
    )
)

final_chain = parallel_chain | custom_output_parser

print(final_chain.get_graph().draw_ascii())

print("\n\n", "--" * 50, "\n\n", sep="")

for response in final_chain.batch(statements):
    print(response)
    print()

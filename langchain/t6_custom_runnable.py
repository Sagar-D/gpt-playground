from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import contractions
import dotenv
import os

dotenv.load_dotenv()


def text_preprocessor(text):

    if not isinstance(text, str):
        raise ValueError("Argument passed should be of string type.")

    # strip spaces
    text = text.strip(" ")

    # Capitalize text
    text = text.capitalize()

    # Remove contractions
    text = contractions.fix(text)

    return text


# Create a runnable for text pre processing
text_preprocessor_runnable = RunnableLambda(text_preprocessor)

# Create a prompt template for sentiment analysis of the texts
sentiment_analysis_template = ChatPromptTemplate.from_template(
    """\
For the below given statment, analyse the sentiment and respond if the sentiment is 'Positive' or 'Negative'
Note : Respond in only one word. Don't add any additional comments or notes.
{statemet}
"""
)

# Create a adapter to concvert pre processor response to format expected by sentiment_analysis_template
prompt_argument_converter = RunnableLambda(lambda text: {"statemet": text})

# Create an llm instance to connect with the LLM model
llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0
)

# create a output parser
parser = StrOutputParser()

# create sentiment analysis chain
sentiment_analysis_chain = (
    text_preprocessor_runnable
    | prompt_argument_converter
    | sentiment_analysis_template
    | llm
    | parser
)


# Sample data set of list of product reviews
product_reviews = [
    "I LOVE this product! It's absolutely amazing.   ",
    "Not bad, but could be better. I've seen worse.",
    "Terrible experience... I'm never buying again!!",
    "Pretty good, isn't it? Will buy again!",
    "Excellent value for the money!!! Highly recommend.",
]

print(sentiment_analysis_chain.batch(product_reviews))

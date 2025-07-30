from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel , Field
import os
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=2)

class City(BaseModel) :
    """Class to hold city details"""

    name: str = Field(description="Name of the city")
    is_capital: bool = Field(description="Is the city capital")
    area: float = Field(description="Land Area of the city in square meters")
    population: int = Field(description="Population of the city")

city_json_parser = JsonOutputParser(pydantic_object=City)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that generates JSON data and responds with JSON data to user.\n Respond only with JSON data and don't add any other text or notes or any code blocks in the response."),
    ("human", """\
Provide the below details of the city - {city_name}
1. Name of the city
2. Is it a capital city?
3. Land area of the city in square kilometers
4. Population of the city
Respond in the format : {response_format}
If you don't know any details, don't include that key in the JSON""")
])

prompt = prompt.partial(response_format=city_json_parser.get_format_instructions())

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"city_name": "Bangalore"}))
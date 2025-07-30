from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import os
import dotenv
from typing import List
from pprint import pprint

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0
)

apollo_story = """
On July 20, 1969, Apollo 11, the first manned mission to land on the Moon, successfully touched down in the Sea of Tranquility. \
The crew consisted of Neil Armstrong, who served as the mission commander, \
Edwin 'Buzz' Aldrin, the lunar module pilot, and Michael Collins, the command module pilot.

The spacecraft consisted of two main parts: the command module Columbia and the lunar module Eagle. \
As Armstrong stepped onto the lunar surface, he famously declared, "That's one small step for man, one giant leap for mankind."

Buzz Aldrin also descended onto the Moon's surface, where he and Armstrong conducted experiments and collected samples. \
Michael Collins remained in lunar orbit aboard Columbia, ensuring the successful return of his fellow astronauts.

The mission was a pivotal moment in space exploration and remains a significant achievement in human history.
"""

# Extract below details from the apollo_story
# 1. all the crew members - (name, role)
# 2. spacecraft parts - (name of the spacecraft, part/module name)
# 3. significant quotes made (quote, speaker)


class CrewMember(BaseModel):
    """Details of a crew member"""

    name: str = Field(description="Name of the crew member")
    role: str = Field(description="Role of the crew member in the mission")


class SpacecraftPart(BaseModel):
    """Details of Spacecraft Parts"""

    name: str = Field(description="Name of the spacecraft")
    part: str = Field(description="Specific part or module of the spacecraft")


class SignificantQuote(BaseModel):
    """Details of Significat quotes"""

    quote: str = Field(description="Quote text")
    speaker: str = Field(description="Speaker who made the quote")


class MissionDetails(BaseModel):
    """Combined details of the Apollo 11 mission"""

    crew_members: List[CrewMember]
    spacecraft_parts: List[SpacecraftPart]
    significant_quotes: List[SignificantQuote]


json_output_parser = JsonOutputParser(pydantic_object=MissionDetails)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI agent who responds only in JSON objects. You follow the instructions provided by the user and provide a JSONresponse based on users request",
        ),
        (
            "human",
            "mission_summary_text : {input}\n Given the above mission_summary_text, extract the requested data based as per below requirement.\n"
            + "\nOutput format instruction : {format_instructions}.",
        ),
    ]
)
prompt_template = prompt_template.partial(
    format_instructions=json_output_parser.get_format_instructions()
)
chain = prompt_template | llm | json_output_parser
pprint(chain.invoke({"input": apollo_story}))

from langchain_ollama import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import os
import dotenv
import requests
from pprint import pprint
from datetime import date
import json

dotenv.load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"), model=os.getenv("LLM_MODEL"), temperature=0
)


class TemperatureDataSchema(BaseModel):
    """Use this tool to get accurate temperature information for a location. Don't use it for general purpose."""

    lattitude: float = Field(..., description="Lattitude of the location")
    longitude: float = Field(..., description="Longitude of the location")
    forcast_date: date = Field(
        description="Date for which temperature data is required"
    )


@tool(args_schema=TemperatureDataSchema)
def temperature_for_location(
    lattitude: float, longitude: float, forcast_date: date = date.today()
):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lattitude,
        "longitude": longitude,
        "daily": "temperature_2m_mean",
        "start_date": forcast_date.strftime("%Y-%m-%d"),
        "end_date": forcast_date.strftime("%Y-%m-%d"),
    }

    response = requests.get(url=url, params=params)
    if response.status_code != 200:
        return f"Rain Forcast is not available for date {forcast_date}"

    response_body = response.json()
    return f"{response_body['daily']['temperature_2m_mean'][0]} {response_body['daily_units']['temperature_2m_mean']}"


class GetAirQualityCategoryForLocation(BaseModel):
    """Use external API to get current and accurate air quality category ('Fair', 'Poor', etc.) for a specified location."""

    latitude: float = Field(..., description="Latitude of the city.")
    longitude: float = Field(..., description="Longitude of the city.")


@tool(args_schema=GetAirQualityCategoryForLocation)
def get_air_quality_category_for_location(latitude, longitude) -> str:
    base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {"latitude": latitude, "longitude": longitude, "hourly": "european_aqi"}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "hourly" in data:
            euro_aqi = data["hourly"]["european_aqi"][0]

            # Determine AQI category
            if euro_aqi <= 20:
                return "Good"
            elif euro_aqi <= 40:
                return "Fair"
            elif euro_aqi <= 60:
                return "Moderate"
            elif euro_aqi <= 80:
                return "Poor"
            elif euro_aqi <= 100:
                return "Very Poor"
            else:
                return "Extremely Poor"
        else:
            return "No air quality data found for the given coordinates."

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"


system_message = """\
You are a helpful assistant capable of tool calling when helpful, necessary, and appropriate.

Think hard about whether or not you need to call a tool, \
based on your tools' descriptions and use them, but only when appropriate!

Whether or not you need to call a tool, address the user's query in a helpful informative way.

You should ALWAYS actually address the query and NEVER discuss your thought process about whether or not to use a tool.
"""

prompt = "'What is the temperature in Mumbai on 1st August 2025?"
agent = create_react_agent(
    llm,
    tools=[temperature_for_location, get_air_quality_category_for_location],
    state_modifier=system_message,
)
agent_state = agent.invoke({"messages": [prompt]})

for message in agent_state["messages"]:
    message.pretty_print()

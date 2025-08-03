from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableAssign
from typing import Optional
import os
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    base_url= os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
    temperature=0
)

database = {
    "rahul_dravid_19" : {
        "first_name" : "Rahul",
        "last_name" : "Dravid",
        "booking_id": 19,
        "departure_from" : "Bangalore",
        "departure_to" : "Chennai",
        "departure_date" : "23-10-2025"
    },
    "sachin_tendulkar_10" : {
        "first_name" : "Sachin",
        "last_name" : "Tendulkar",
        "booking_id": 10,
        "departure_from" : "Mumbai",
        "departure_to" : "Bangalore",
        "departure_date" : "24-10-2025"
    },
    "virat_kohli_18" : {
        "first_name" : "Virat",
        "last_name" : "Kohli",
        "booking_id": 18,
        "departure_from" : "Delhi",
        "departure_to" : "Bangalore",
        "departure_date" : "23-10-2025"
    },
    "rohit_sharma_45" : {
        "first_name" : "Rohit",
        "last_name" : "Sharma",
        "booking_id": 45,
        "departure_from" : "Mumbai",
        "departure_to" : "Chennai",
        "departure_date" : "20-10-2025"
    } 
}

def fetch_data(knowledge_base) :
    if (not knowledge_base['first_name']) or (not knowledge_base['last_name']) or (not knowledge_base['booking_id']) :
        return ""
    
    first_name = knowledge_base['first_name']
    last_name = knowledge_base['last_name']
    booking_id = knowledge_base['booking_id']
    key = f"{first_name.lower()}_{last_name.lower()}_{booking_id}"
    if key in database :
        return database[key]
    return f"No flight booking found for customer {first_name} {last_name} with booking id {booking_id}."
    
class KnowledgeBase(BaseModel) :
    """Hold the chat context and knowledge base"""
    first_name: Optional[str] = Field("unkown", description="First name of the user, unknown if not known")
    last_name: Optional[str] = Field("unkown", description="Last name of the user, unknown if not known")
    booking_id: Optional[int] = Field(0, description="Booking Confirmation Id of air ticket, empty if not known")
    chat_summary: Optional[str] = Field("", description="Summary of the conversation with user")

knowledge_output_parser = JsonOutputParser(pydantic_object=KnowledgeBase)

knowledge_prompt = ChatPromptTemplate.from_template("""\
You are knowledge base data updation system.
Respond back with the updated KNOWLEDGE BASE using the latest 'input' given by user.
OLD KNOWLEDGE BASE : {knowledge_base}
INPUT : {user_input}
Respond using below format : 
{format_instruction}\
""")

knowledge_prompt = knowledge_prompt.partial(format_instruction=knowledge_output_parser.get_format_instructions())

knowledge_chain = knowledge_prompt | llm | knowledge_output_parser


CHATBOT_SYSTEM_INSTRUCTION = """\
You are an customer assisstant chatbot for Indigo Airlines. You are job is to understand user queries and help them with right information.

Indigo airlines adheres to the general aviation guidelines followed by all the standard Aviation Companies. Answer your queries keeping that in mind.

Here is your personal knowledge base on customer details : {knowledge_base}
This knowledge base is only for your reference. Don't share the details from this to customere, unless required.

Based on the customer details in the above knowledge_base, here is some customer flight booking context for your reference : {booking_context}
Use these details for assisting the customer whenever neccessary.

Keep in mind, to fetch flight booking context of user we need
1. First name of the customer
2. Last name of the customer
3. Booking Id

Keep your coversations short and sweet. Avoid unneccsary greetings or other chats which aren't useful.
Important Note : Do not hallucinate! . If you don't have any information or if you can't perform any task, acknowledge it.
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CHATBOT_SYSTEM_INSTRUCTION),
        ("placeholder", "{chat_history}")
    ]
)

class FlightBookingInfo(BaseModel) :
    """Use this tool to fetch the details of customer flight bookings and their departure and arrival information"""
    first_name: str = Field(..., description="First name of the user")
    last_name: str = Field(..., description="Last name of the user")
    confirmation_id: str = Field(..., description="Booking Reference Id of air ticket")

chat_chain = chat_prompt | llm | StrOutputParser()

state_chain = (
    RunnableAssign({"knowledge_base": knowledge_chain, "abc" : lambda x : "new value"})
    | RunnableAssign({"booking_context": lambda state : fetch_data(state['knowledge_base'])})
)



chat_history = []
state = {
    'knowledge_base': KnowledgeBase(),
    'booking_context': "Not available",
    'user_input': "",
    'chat_history': chat_history
}

while True :
    user_prompt = input("\n\n[Customer] : ")
    chat_history.append(("human",user_prompt))
    state['user_input'] = user_prompt
    state = state_chain.invoke(state)
    print("--"*40)
    print(state['knowledge_base'])
    print(state['booking_context'])
    print("--"*40)
    bot_response = chat_chain.invoke(state)
    print(f"\n\n[Assistant] : {bot_response}\n")
    chat_history.append(("ai", bot_response))
    

"""
Author: Rajendhiran Easu
Date: 17/01/26
Description: 
"""

import requests
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

VALID_CATEGORIES = ["Any", "Misc", "Programming", "Dark", "Pun", "Spooky", "Christmas"]


@tool("tell_me_a_joke",
      description="""Fetches a joke from JokeAPI.
        Args:
            category: MUST be EXACTLY one of these: Any, Misc, Programming, Dark, Pun, Spooky, Christmas
                     Use 'Any' as fallback if user request doesn't match any category.
            
        Returns:
            A joke string
        """)
def tell_me_a_joke(category: str):
    url = f"https://v2.jokeapi.dev/joke/{category if category in VALID_CATEGORIES else "any"}?type=single"
    print(url)
    response = requests.get(url)
    print(response.text)
    return response.json()


@tool("get_weather", description="return a weather details on the provided city")
def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=j1"
    print(url)
    res = requests.get(url)
    print(res.text)
    return res.json


llm = ChatOllama(model="llama3.2:latest", temperature=0)
ai_agent = create_agent(
    model=llm,
    tools=[tell_me_a_joke, get_weather],
    system_prompt="""You are a very intelligent AI works on telling weather report and crack jokes. 
    For Jokes your ONLY job is to:
        1. Use the tell_me_a_joke tool to fetch jokes
        2. Return EXACTLY what the tool gives you
        3. NEVER make up your own jokes

        When user asks for a joke, call the tool with the appropriate category.
    For Weather report call a `get_weather` with the respective user city and provide the necessary details.
        """
)
#
# # result = ai_agent.invoke({
# #     "messages": [
# #         {"role": "user",
# #          "content": "Tell me what kinda joke you want: "}
# #     ]
# # })
#
user_input = input("Tell me what kinda joke or weather info of city you want: ")
result = ai_agent.invoke({
    "messages": [HumanMessage(content = user_input.strip())]
})
# print(result["messages"])
print(result["messages"][-1].content)

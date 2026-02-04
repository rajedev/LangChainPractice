"""
Author: Rajendhiran Easu
Date: 25/01/26
Description: 
"""

## Here to check, how the internal tool messages works with tool id

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

system_message = SystemMessage("You are an intelligent bot assistant can use the tool calling to find and share the weather details to the user for users locality")
user_message = HumanMessage("what is the weather in puducherry?")
ai_message = AIMessage(content=[], tool_calls=[{
    "name": "get_weather",
    "args": {"location": "puducherry"},
    "id": "weat_123"
}])

tool_message = ToolMessage(content="summer but rainy with cold of 20C",
                           tool_call_id="weat_123")

llm = ChatOllama(model="llama3.2:latest", temperature=0)

msg = [system_message, user_message, ai_message, tool_message]

response = llm.invoke(msg)
print(response)

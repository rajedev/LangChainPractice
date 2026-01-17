"""
Author: Rajendhiran Easu
Date: 17/01/26
Description: 
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:latest", temperature=0.7)
ai_agent = create_agent(
    model=llm,
    system_prompt="You are an ai assistant expert on coding best practice"
)

# result = ai_agent.invoke({
#     "messages": [
#         {"role": "user",
#          "content": "tell me 2 best practice with kotlin code?"}
#     ]
# })

user_input = input("Question Pls: ")
result = ai_agent.invoke({
    "messages": [HumanMessage(user_input.strip())]
})
# print(result["messages"])
print(result["messages"][-1].content)

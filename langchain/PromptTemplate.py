"""
Author: Rajendhiran Easu
Date: 10/01/26
Description: 
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2:latest", temperature=0.7)

# Prompt Template

# prompt = ChatPromptTemplate.from_template("Tell me a joke on {user_input}?")
# prompt = ChatPromptTemplate.from_messages([
#     ("ai", "generate a mock details in the prescribed {format}"),
#     ("user", "provide a details on {instruction}")
# ])

prompt = ChatPromptTemplate.from_messages([
    AIMessagePromptTemplate.from_template("generate a mock details in the prescribed {format}"),
    HumanMessagePromptTemplate.from_template("provide a details on {instruction}")
])

# Check prompt value
# prompt_value = prompt.invoke({"format": "json",
#                         "instruction": "3 to 5 simple student details who are studying in different dept. in the college"})
# print(prompt_value)
# response = model.invoke(prompt_value)
# print(response.content)

# Constructing the chain (LCEL)
chain = prompt | model
response = chain.invoke({"format": "json",
                         "instruction": "3 to 5 simple employee details who are working in different dept. in the factory"})
print(response.content)

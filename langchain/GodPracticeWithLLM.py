"""
Author: Rajendhiran Easu
Date: 10/01/26
Description: 
"""

from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model

llm = init_chat_model(model="llama3.2:latest", model_provider="ollama", temperature=0.7)

#llm = ChatOllama(model="llama3.2:latest", temperature=0.7)
result = llm.invoke("Tell me something on learning langchain? in 1 simple line")
# result = llm.batch(["simple tips on learning in 1 line", "langchain?", "langraph?"])
# result = llm.stream(["simple tips on learning in 10 words", "langchain?"])

print(result.content)

# for msg in result:
#     print(msg.content, end="\n\n")
#     #print(msg.model_dump_json(indent=2),end="\n\n", flush=True)

# for msg in result:
#     print(msg.content, end="", flush=True)

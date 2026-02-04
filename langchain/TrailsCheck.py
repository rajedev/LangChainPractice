"""
Author: Rajendhiran Easu
Date: 25/01/26
Description: 
"""

from langchain_ollama import ChatOllama, OllamaLLM

llm = ChatOllama(model="llama3.2:latest", temperature=0.3)

##
#llm = OllamaLLM(model="llama3.2:latest", temperature=0.3)

result = llm.invoke("Pointing to a photograph, a man said, “I have no brother or sister but that man’s father is my father’s son.” Whose photograph was it? ")

# string result on text completion
#print(result)

#Chat
print(result.content)
print(result.response_metadata)
print(result.usage_metadata)
print(result.id)
print(result.content_blocks)

#streaming output
#for response in result:
    # print(response.content)
    # print(response.response_metadata)
    # print(response.usage_metadata)
    # print(response.id)
    # print(response.content_blocks)

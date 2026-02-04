"""
Author: Rajendhiran Easu
Date: 26/01/26
Description: 
"""
from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(model="embeddinggemma:300m")

vector_data_single_query = embedding_model.embed_query("God Knows All, Trust in God")

print(len(vector_data_single_query))
print(vector_data_single_query[:3])

data_list = [
    "Welcome to AI",
    "Python is primary here",
    "LangChian, LangGraph is a framework"
]

vector_data_documents = embedding_model.embed_documents(data_list)
print(len(vector_data_documents))

for doc in vector_data_documents:
    print(len(doc))
    print(doc[:5]) ## first 5 vector data

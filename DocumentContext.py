"""
Author: Rajendhiran Easu
Date: 10/01/26
Description: 
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2:latest", temperature=0.2)
doc = Document("""Tourist Family is a 2025 Indian Tamil comedy-drama film written and directed by Abishan Jeevinth in his directorial debut.
Produced by Million Dollar Studios and MRP Entertainment, the film stars M. Sasikumar as Dharmadas “Das” and Simran as Vasanthi, his wife.
Yogi Babu plays Prakash, Vasanthi’s elder brother, while Ramesh Thilak appears as A. Bhairavan.
The supporting cast includes M. S. Bhaskar as Richard, Elango Kumaravel as Gunasekar, Sreeja Ravi as Mangaiyarkarasi, and Bagavathi Perumal as Inspector R. Raghavan.""")

## we can also add the context content directly here to run and see the output.
prompt = ChatPromptTemplate.from_template(
    """
    Your are an AI Assistant answer the user questions
    Question: {user_input}
    Context: {information}
    """
)

chain = prompt | model
response = chain.invoke({
    "user_input":input("Question: "),
    "information":[doc]
})
print(response.content)

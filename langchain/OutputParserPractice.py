"""
Author: Rajendhiran Easu
Date: 10/01/26
Description: 
"""
import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ConfigDict

model = ChatOllama(model="llama3.2:latest", temperature=0.5)


## String output parser
# prompt = ChatPromptTemplate.from_messages([
#     ("system","Provide a 2 pros and cons on the user input"),
#     ("user","{user_input}")])
#
# parser = StrOutputParser()
# chain = prompt | model | parser
# response = chain.invoke({
#     "user_input":"ecommerce"
# })
# print(response)

## CSV (list) output parser
# prompt = ChatPromptTemplate.from_messages([
#     ("system","Provide a list of prerequisite skillsets to learn the below user input, keep the list simple to 5 to 8 words for each and can list item also 5 to 7 max."),
#     ("user","{user_input}")])

# parser = CommaSeparatedListOutputParser()
# chain = prompt | model | parser

# user_input =input("Enter the title of your learning topic to know the list of prerequisites?")
# response = chain.invoke({
#     "user_input":user_input
# })
# print(response)

## JSON - Pydantic output parser
class Casting(BaseModel):
    actor: str
    character: str


class MovieInfo(BaseModel):
    model_config = ConfigDict(validate_by_name=True, extra="forbid")
    name: str = Field(alias="movie_name", description="The name of the movie")
    genre: list[str] = Field(description="Movie genres")
    release_year: int = Field(description="The year the movie was released")
    produced_by: list[str] = Field(description="The producer of the movie")
    directed_by: str = Field(description="The Director of the movie")
    crew_members: list[str] = Field(description="all actor names")
    movie_casting: list[Casting] = Field(
        description="List of actor and cast mappings")


prompt = ChatPromptTemplate.from_messages([
    # ("system", "Extract information from the movie details and provide the formated info {instruction}"),
    ("system",
     "You are an information extraction assistant."
     "Do NOT include schema definitions, field descriptions,"
     "model_config, validation rules, or any extra keys."
     "Extract movie details from the text and output ONLY valid JSON "
     "that matches the given schema.\n{instruction}"),
    ("user", "{movie_details}")
])

parser = JsonOutputParser(pydantic_object=MovieInfo)
chain = prompt | model | parser
# chain = RunnableSequence(prompt, model, parser)
# print(parser.get_format_instructions())
response = chain.invoke({
    "movie_details": """Tourist Family is a 2025 Indian Tamil comedy-drama film written and directed by Abishan Jeevinth and produced by Million Dollar Studios and MRP Entertainment.
The film stars M. Sasikumar as Dharmadas and Simran as Vasanthi, with Yogi Babu as Prakash and Ramesh Thilak as A. Bhairavan.
The cast includes M. S. Bhaskar as Richard, Elango Kumaravel as Gunasekar, Sreeja Ravi as Mangaiyarkarasi, and Bagavathi Perumal as Inspector R. Raghavan.""",
    "instruction": parser.get_format_instructions(),
})

print(response)
print(type(response))
json_data = json.dumps(response, indent=2)
print(json_data)

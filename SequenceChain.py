"""
Author: Rajendhiran Easu
Date: 15/01/26
Description:
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

MODEL_NAME = "llama3.2:latest"
MODEL_TEMPERATURE = 0.7
DEFAULT_ARTICLE_COUNT = 2
DEFAULT_TONE = "Polite"

article_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an ai assistant to simple articles on {topic}"),
    ("user", "limits to 100 words on the descriptions in the {tone_type} tone")
])

linkedin_post_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a social media expert, Help to rewrite the article into a viral linkedin post"),
    ("user", "Summary: {content}")
])


def get_user_input_for_article() -> tuple[str, str]:
    topic = input("Article Topic: ")
    tone = input(f"Briefing Tone Type - (Default: {DEFAULT_TONE}): ").strip() or DEFAULT_TONE
    return topic, tone


def data_for_linkedin_post(description: str) -> dict[str, str]:
    print(f"Description: {description}")
    return {"content": description}


# data_for_linked_in = lambda desc: {"content": desc}


data_for_linked_in = RunnableLambda(data_for_linkedin_post)


def generate_linkedin_post() -> str:
    topic, tone = get_user_input_for_article()
    llm = ChatOllama(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)
    chain = article_prompt | llm | StrOutputParser()
    linkedin_chain = linkedin_post_prompt | llm | StrOutputParser()
    full_chain = chain | data_for_linked_in | linkedin_chain

    return full_chain.invoke(
        {"topic": topic,
         "tone_type": tone
         })


if __name__ == "__main__":
    result = generate_linkedin_post()
    print(type(result))
    print(result)

"""
Author: Rajendhiran Easu
Date: 15/01/26
Description:
"""
from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import AfterValidator
from pydantic import BaseModel, Field

MODEL_NAME = "llama3.2:latest"
MODEL_TEMPERATURE = 0.7
DEFAULT_ARTICLE_COUNT = 2
DEFAULT_TONE = "Polite"


def _ensure_hashtag(tags: list[str]) -> list[str]:
    return [tag.strip() if tag.startswith("#") else f"#{tag.strip()}" for tag in tags]


# _ensure_hashtag_anony = lambda tags :[tag.strip() if tag.startswith("#") else f"#{tag.strip()}" for tag in tags]

class Article(BaseModel):
    title: str = Field(description="title of the article")
    description: str = Field(description="description of the article")
    tags_name: Annotated[list[str], Field(description="list of hashtags relates to the article title"), AfterValidator(
        _ensure_hashtag)]


class ArticlesColumn(BaseModel):
    articles: list[Article] = Field(description="list of articles")


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an ai assistant to provide {no_of_article} special articles on {topic}"),
    ("user", "limits to 25 words on each descriptions and hashtags of min. 3 to 5 in the {tone_type} tone")
])


def get_user_input() -> tuple[str, int, str]:
    topic = input("Article Topic: ")
    try:
        count = int(input(f"No. of article - (Default:{DEFAULT_ARTICLE_COUNT}): "))
    except ValueError as _:
        count = DEFAULT_ARTICLE_COUNT

    tone = input(f"Briefing Tone Type - (Default: {DEFAULT_TONE}): ").strip() or DEFAULT_TONE
    return topic, count, tone


def generate_article() -> ArticlesColumn:
    topic, no_of_article, tone = get_user_input()
    llm = ChatOllama(model=MODEL_NAME, temperature=MODEL_TEMPERATURE).with_structured_output(schema=ArticlesColumn)
    # chain = RunnableSequence(prompt, llm)
    chain = prompt | llm

    return chain.invoke(
        {"topic": topic,
         "tone_type": tone,
         "no_of_article": no_of_article
         })


if __name__ == "__main__":
    result = generate_article()
    print(type(result))
    print(result)
    print(result.model_dump_json(indent=2))

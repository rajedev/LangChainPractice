"""
Author: Rajendhiran Easu
Date: 15/01/26
Description: 
"""
from functools import wraps
from typing import Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


def log_info(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function Call'd: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


MODEL_NAME = "llama3.2:latest"
MODEL_TEMPERATURE = 0.7

model = ChatOllama(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI expert to analysis the provided news content and act upon the user request"),
    ("user",
     "Helps to determine the news headline: {news_headline}, related to which category (Politics, Sports, Spiritual, Crime. In case if that headline not fits into anything, just tag it to the `common` category)")
])


@log_info
def content_for_front_page_news(news_headline: str):
    news_paper_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the news reporter expert, update the provided news for the newspaper front page as much as informative"),
        ("user", "{news}")
    ])
    chain = news_paper_prompt | model | StrOutputParser()
    return chain.invoke({"news": news_headline})


@log_info
def content_for_instagram(feed: str):
    insta_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the instagram influencer expert, make this content for the instagram attractive"),
        ("user", "{feed}")
    ])
    chain = insta_prompt | model | StrOutputParser()
    return chain.invoke({"feed": feed})


@log_info
def content_for_youtube_description(details: str):
    youtube_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the youtube content creator, make this content for the youtube post description"),
        ("user", "{description}")
    ])
    chain = youtube_prompt | model | StrOutputParser()
    return chain.invoke({"description": details})


@log_info
def content_for_whatsapp_msg(details: str):
    whatsapp_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the whatsapp simple message generator, make this content for the simple whatsapp msg"),
        ("user", "{msg}")
    ])
    chain = whatsapp_prompt | model | StrOutputParser()
    return chain.invoke({"msg": details})


# whatsapp_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are the whatsapp simple message generator, make this content for the simple whatsapp msg"),
#         ("user", "{msg}")
#     ])

class NewsData(BaseModel):
    headline: Annotated[str, Field(description="news headline or summary")]
    category: Annotated[str, Field(description="news category")] = ""


branch_category = RunnableBranch(
    (
        lambda feed: "Politics" in feed.category,
        RunnableLambda(lambda feed: content_for_front_page_news(feed.headline))
    ),
    (
        lambda feed: feed.category in ["Spiritual", "Crime"],
        RunnableLambda(lambda feed: content_for_youtube_description(feed.headline))
    ),
    (
        lambda feed: "Sports" in feed.category,
        RunnableLambda(lambda feed: content_for_instagram(feed.headline))
    ),
    # whatsapp_prompt | model | StrOutputParser()
    RunnableLambda(lambda feed: content_for_whatsapp_msg(feed.headline))
)


@log_info
def check_news_category():
    llm = ChatOllama(model=MODEL_NAME, temperature=MODEL_TEMPERATURE).with_structured_output(schema=NewsData)
    category_chain = prompt | llm | branch_category
    input_news = input("News Headline Pls: ")
    return category_chain.invoke({"news_headline": input_news})


if __name__ == "__main__":
    result = check_news_category()
    print(type(result))
    print(result)

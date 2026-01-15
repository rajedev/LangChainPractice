"""
Author: Rajendhiran Easu
Date: 15/01/26
Description:
"""
from typing import Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
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


llm_general = ChatOllama(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)


def get_user_input() -> tuple[str, int, str]:
    topic = input("Article Topic: ")
    try:
        count = int(input(f"No. of article - (Default:{DEFAULT_ARTICLE_COUNT}): "))
    except ValueError as _:
        count = DEFAULT_ARTICLE_COUNT
    count = max(DEFAULT_ARTICLE_COUNT, count)
    tone = input(f"Briefing Tone Type - (Default: {DEFAULT_TONE}): ").strip() or DEFAULT_TONE
    return topic, count, tone


def content_for_social_media_post(articles_column: ArticlesColumn):
    article = articles_column.articles[0]
    print(f"Article: {article}")
    return {"subject": article.title, "content": article.description}


def article_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an ai assistant to provide {no_of_article} special articles on {topic}"),
        ("user", "limits to 25 words on each descriptions and hashtags of min. 3 to 5 in the {tone_type} tone")
    ])
    llm = ChatOllama(model=MODEL_NAME, temperature=MODEL_TEMPERATURE).with_structured_output(schema=ArticlesColumn)
    chain = prompt | llm
    return chain


def linkedin_chain():
    linkedin_post_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a social media expert. Help to rewrite the article summary into viral linkedin post"),
        ("user", "title:{subject} ; summary:{content}")
    ])

    linkedin_post_chain = linkedin_post_prompt | llm_general | StrOutputParser()

    return linkedin_post_chain


def instagram_chain(content_dict: dict[str, str]):
    input_content = content_dict["content"]
    print(f"Insta: {input_content}")
    insta_post_prompt = ChatPromptTemplate.from_messages([(
        "system", "You are a social media expert. Help to rewrite the article summary into viral insta post"),
        "user", "influence my viewer on: {content}"
    ])
    insta_post_chain = insta_post_prompt | llm_general | StrOutputParser()
    return insta_post_chain.invoke({"content": input_content})


# Sequential Chain
def generate_linkedin_post():
    topic, no_of_article, tone = get_user_input()
    output_chain = (article_chain() |
                    RunnableLambda(content_for_social_media_post) |
                    linkedin_chain())
    return output_chain.invoke(
        {"topic": topic,
         "tone_type": tone,
         "no_of_article": no_of_article
         })


##Parallel Chain
def generate_social_media_post():
    topic, no_of_article, tone = get_user_input()
    insta_runnable = RunnableLambda(instagram_chain)
    output_chain = (article_chain() |
                    RunnableLambda(content_for_social_media_post) |
                    RunnableParallel(branch={"linkedin": linkedin_chain(), "insta": insta_runnable}) |
                    RunnableLambda(display_post))

    return output_chain.invoke(
        {"topic": topic,
         "tone_type": tone,
         "no_of_article": no_of_article
         })


def display_post(social_media: dict[str, str]):
    msg_post = social_media["branch"]
    __beautify(post=msg_post["linkedin"], title="LinkedIn Post")
    __beautify(post=msg_post["insta"], title="Instagram Post")


def __beautify(post: str, title: str):
    print("*" * 15)
    print(title)
    print("*" * 15)
    print(post, end="\n\n")


if __name__ == "__main__":
    generate_social_media_post()
    ## Linkedin post with pydantic object
    ## Sequential Chain
    # result = generate_linkedin_post()
    # print(type(result))
    # print(result)

    ## Parallel Chain
    # result = generate_social_media_post()
    # print(type(result))
    # json_data = json.dumps(result, indent=2)
    # print(json_data)
    # branch_result = result["branch"]
    # print(branch_result["linkedin"])
    # print("#" * 100)
    # print(branch_result["insta"])

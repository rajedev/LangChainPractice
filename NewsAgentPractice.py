"""
Author: Rajendhiran Easu
Date: 20/01/26
Description: 
"""

import requests
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage


@tool("get_news", description="""Get all news posts from the feed.
    
    Returns:
        List of all news posts with their categories
    """)
def get_news() -> list:
    url = "https://my-json-server.typicode.com/rajendhirandev/mock_api_feed/myWall"
    try:
        response = requests.get(url, timeout=5)
        all_posts = response.json()
        print(f"✅ Fetched {len(all_posts)} posts")
        return all_posts

    except Exception as e:
        return [{"error": f"Failed to fetch news: {str(e)}"}]


model = init_chat_model(model="llama3.2:latest", model_provider="ollama", temperature=0.0)

agent = create_agent(
    model=model,
    system_prompt="""You are a news assistant that helps users find posts by category.
                INSTRUCTIONS:
                1. Call the get_news tool to fetch all news posts
                2. From the tool response, filter and show ONLY posts matching the user's requested category
                3. If user doesn't specify a category, use "General"
                4. If no posts match the category, use "General"
                5. Present the filtered posts clearly
            Each post has a 'category' field - use it to filter the results.""",
    tools=[get_news]
)

user_input = input("Which category news do you want? (e.g., Tech, Sports, General): ")
result = agent.invoke(
    {
        "messages": [HumanMessage(user_input.strip())]
    }
)
print("\n" + "=" * 60)
print("NEWS RESULTS:")
print("=" * 60)
print(result["messages"][-1].content)
print("=" * 60)

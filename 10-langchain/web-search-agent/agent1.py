from dotenv import load_dotenv
import os

load_dotenv()
from netfree_unstrict_ssl import unstrict_ssl

unstrict_ssl()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import format_message
from firecrawl import Firecrawl
from langchain_core.tools import tool

_model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

_firecrawl = Firecrawl()

@tool
def firecrawl_search(query: str) -> list[dict]:
    """Search the web for a topic and return the most relevant sources (title + URL)."""
    result = _firecrawl.search(query, limit=3)
    return [{"title": item.title, "url": item.url}
            for item in (result.web or [])]
   
agent = create_agent(
    model=_model,
    system_prompt="""
You are a careful web researcher.

For every user question:
1. Rewrite the user's question into a clear search query.
2. Use Firecrawl Search to find relevant sources.
3. Answer only based on the sources.
4. If the question is ambiguous, explain the ambiguity and answer the most likely interpretation.
5. Include source URLs in the final answer.
""",
    tools=[firecrawl_search],
)


def stream_query(user_input: str):
    for step in agent.stream({"messages": user_input}, stream_mode="values"):
        message = step["messages"][-1]
        yield format_message(message=message)
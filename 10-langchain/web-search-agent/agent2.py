from dotenv import load_dotenv

load_dotenv()
from netfree_unstrict_ssl import unstrict_ssl

unstrict_ssl()

import uuid
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents.middleware import SummarizationMiddleware

from utils import format_message, conversation_usage
from firecrawl import Firecrawl
from pydantic import BaseModel, Field

_model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")


# ======TOOLS======
_firecrawl = Firecrawl()

@tool
def firecrawl_search(query: str) -> list[dict]:
    """Search the web for a topic and return the most relevant sources (title + URL)."""
    result = _firecrawl.search(query, limit=3)
    return [{"title": item.title, "url": item.url}
            for item in (result.web or [])]

@tool
def firecrawl_scrape(url: str) -> str:
    """Fetch the full content of a source URL as clean markdown."""
    result = _firecrawl.scrape(url, formats=["markdown"], only_main_content=True)
    return result.markdown or "No content scraped"

@tool
def firecrawl_crawl(url: str, limit: int = 5) -> list[dict]:
    """Crawl a website and its accessible subpages, returning the markdown content
    of each page (up to `limit` pages) together with its source URL."""
    result = _firecrawl.crawl(url, limit=limit, formats=["markdown"])
    return [{"url": doc.metadata.url if doc.metadata else url, "markdown": doc.markdown or ""}
            for doc in (result.data or [])]

# ======END TOOLS======

# ======SUTRCTURED OUTPUT======

class ResearchAnswer(BaseModel):
    """Structured answer grounded only in the selected sources."""
    answer: str = Field(description="The answer, based only on the selected sources")
    sources_used: list[str] = Field(
        description="URLs of the selected sources actually used for this answer"
    )
    answered_from_sources: bool = Field(
        description="False if the question cannot be answered from the selected sources"
    )

# ======END SUTRCTURED OUTPUT======

agent2 = create_agent(
    model=_model,
    system_prompt="""
You are a research assistant that helps users explore a topic in depth.

Workflow:
1. When the user provides a topic, use Firecrawl Search to find sources, pick a few of the most relevant ones, and return the list of selected sources (titles + URLs) to the user.
2. For any follow-up question, answer based only on the content of those selected sources. Use Firecrawl Crawl to fetch the full content of a source when you need more detail than the search snippet provides.
3. Always cite the source URLs you used in your answer.
4. If a question cannot be answered from the selected sources, say so clearly.
""",
    tools=[firecrawl_search, firecrawl_scrape, firecrawl_crawl],
    checkpointer=InMemorySaver(),
    response_format=ResearchAnswer,
    middleware=[
        SummarizationMiddleware(
            model=_model,
            trigger=("tokens", 3000),
            keep=("messages", 5),
        ),
    ],
)


def stream_query(user_input: str, id: str):
    final_step = None
    for step in agent2.stream({"messages": user_input},
                              {"configurable": {"thread_id": id}},
                              stream_mode="values"):
        final_step = step
        formatted = format_message(message=step["messages"][-1])
        if formatted:
            yield formatted

    # The structured output lives under its own state key, not in `messages`.
    structured = (final_step or {}).get("structured_response")
    if structured is not None:
        yield "answer", structured.model_dump()

    # Token accounting + whether SummarizationMiddleware compressed the history.
    yield "usage", conversation_usage((final_step or {}).get("messages", []))
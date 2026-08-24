from dataclasses import dataclass

from firecrawl import Firecrawl
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

from core.config import MODEL
from core.store import SourceStore, store
from core.sources import format_docs

@dataclass
class Answer:
    text: str
    sources: list[str]


SYSTEM_PROMPT = "You are the assistant for a notebook of source documents"

_firecrawl = Firecrawl()


def _make_tools(store: SourceStore):

    @tool
    def search_sources(query: str) -> str:
        """Find passages in the active sources that are relevant to a query"""
        docs = store.search(query=query)
        if not docs:
            return "No relevant documents found in the active sources"
        return format_docs(docs)

    @tool
    def list_sources() -> str:
        """List the names of the active sources, without searching their content.
        Use this when the user asks what sources/documents are available."""
        active_ids = store.active_ids()
        names = [s.name for s in store.list() if s.id in active_ids]
        if not names:
            return "No active sources."
        return "\n".join(names)

    @tool
    def get_source_by_name(name: str) -> str:
        """Return the full content of one active source, given its exact name.
        Use this when the user asks about a specific named source rather than a topic to search."""
        active_ids = store.active_ids()
        for source in store.list():
            if source.id in active_ids and source.name == name:
                return source.content
        return f"No active source named '{name}'. Use list_sources to see available names."

    @tool
    def web_search(query: str) -> list[dict]:
        """Search the public web for a topic and return the most relevant sources
        (title + URL). Use this when the answer isn't in the notebook's sources."""
        result = _firecrawl.search(query, limit=3)
        return [{"title": item.title, "url": item.url} for item in (result.web or [])]

    @tool
    def web_scrape(url: str) -> str:
        """Fetch the full content of a single web page as clean markdown."""
        result = _firecrawl.scrape(url, formats=["markdown"], only_main_content=True)
        return result.markdown or "No content scraped"

    @tool
    def web_crawl(url: str, limit: int = 5) -> list[dict]:
        """Crawl a website and its accessible subpages, returning the markdown content
        of each page (up to `limit` pages) together with its source URL."""
        result = _firecrawl.crawl(url, limit=limit, formats=["markdown"])
        return [
            {"url": doc.metadata.url if doc.metadata else url, "markdown": doc.markdown or ""}
            for doc in (result.data or [])
        ]

    return [
        search_sources,
        list_sources,
        get_source_by_name,
        web_search,
        web_scrape,
        web_crawl,
    ]


# Built once, at import: the checkpointer holds the conversation history in memory, so a
# new one per request would give every turn an empty thread.
_agent = create_agent(
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
    tools=_make_tools(store),
)


def answer(question: str, thread_id: str) -> Answer:
    config = {"configurable": {"thread_id": thread_id}}
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config=config
    )

    text = result["messages"][-1].text
    # TODO add sources to answer
    return Answer(text=text, sources=[])
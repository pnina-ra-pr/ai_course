from dotenv import load_dotenv

load_dotenv()
from netfree_unstrict_ssl import unstrict_ssl

unstrict_ssl()

from langchain.agents import create_agent

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model

from firecrawl import Firecrawl
from langchain_core.tools import tool

# model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
model = init_chat_model("google_genai:gemini-3.1-pro-preview")

_firecrawl = Firecrawl()


@tool
def firecrawl_search(query: str) -> list[dict]:
    """Search the web for a topic and return the most relevant sources (title + URL)."""
    result = _firecrawl.search(query, limit=3)
    return [{"title": item.title, "url": item.url} for item in (result.web or [])]


agent = create_agent(
    model=model,
    system_prompt="""You are a web researcher""",
    tools=[firecrawl_search],
)

user_input = "How much AI Agents live in world right now?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    stream_mode="values",
):
    message = step["messages"][-1]
    print(message.model_dump_json(indent=2))

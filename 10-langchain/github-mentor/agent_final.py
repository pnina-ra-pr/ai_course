from netfree_unstrict_ssl import unstrict_ssl

unstrict_ssl()
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from print_messages import print_messages, empty_usage, report_usage
from middlewares import clear_tool_outputs

import os
import re
import asyncio

REPO_URL_RE = re.compile(
    r"github\.com[:/]([^/\s]+)/([^/\s#?]+?)(?:\.git)?(?:[/\s#?]|$)"
)


def parse_repo(text: str) -> tuple[str, str] | None:
    m = REPO_URL_RE.search(text)
    if not m:
        return None
    return m.group(1), m.group(2)


def build_system_prompt(owner: str, repo: str) -> str:
    return (
        f"You are a GitHub mentor helping the user learn from the open-source "
        f"repository **{owner}/{repo}**. Every user "
        f"question refers to this repo. When calling GitHub MCP tools, always "
        f'pass owner="{owner}" and repo="{repo}" — do not ask which repo. '
        f"Explain code, architecture, history, issues, and PRs in a teaching "
        f"style: surface *why* things are done, not just *what*. Read files, "
        f"commits, and discussions via the MCP tools before answering when "
        f"the answer depends on repo content."
    )


async def prompt_for_repo() -> tuple[str, str]:
    while True:
        first = input("Paste a GitHub repo URL to study: ").strip()
        if first.lower() in {"exit", "quit", ""}:
            raise SystemExit(0)
        parsed = parse_repo(first)
        if parsed:
            return parsed
        print("Could not find a github.com URL in that input — try again.")


async def main():
    client = MultiServerMCPClient(
        {
            "github": {
                "transport": "http",
                "url": "https://api.githubcopilot.com/mcp/",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
                    "X-MCP-Toolsets": "repos,issues",
                    "X-MCP-Readonly": "true"
                },
            }
        }
    )

    mcp_tools = await client.get_tools()
    model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

    owner, repo = await prompt_for_repo()
    print(f"Locked in: {owner}/{repo}. Ask anything about this repo (exit/quit to leave).\n")

    agent = create_agent(
        model,
        tools=mcp_tools,
        system_prompt=build_system_prompt(owner, repo),
        middleware=[clear_tool_outputs],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "session-1"}}
    session_usage = empty_usage()
    prev_len = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        new_msgs = [
            m for m in result["messages"][prev_len:]
            if type(m).__name__ != "HumanMessage"
        ]
        prev_len = len(result["messages"])

        print_messages(new_msgs)

        report_usage(new_msgs, session_usage)


if __name__ == "__main__":
    asyncio.run(main())
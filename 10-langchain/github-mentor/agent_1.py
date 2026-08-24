from netfree_unstrict_ssl import unstrict_ssl

unstrict_ssl()
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from print_messages import print_messages

import os
import asyncio

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

    agent = create_agent(model, tools=mcp_tools)

    while True:
        user_query = input("Enter your query (or 'exit' to quit): ")
        if user_query.strip().lower() in ("exit", "quit"):
            break

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_query,
                    }
                ]
            }
        )

        print_messages(result["messages"])

if __name__ == "__main__":
    asyncio.run(main())
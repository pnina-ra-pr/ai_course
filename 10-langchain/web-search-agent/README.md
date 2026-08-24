# Web Research Agent

A Streamlit chat app powered by a LangChain agent that uses Gemini for reasoning and Tavily for web search.

- [agent.py](agent.py) — the LangChain agent (`stream_query`) wrapping Gemini + Tavily.
- [app.py](app.py) — the Streamlit chat UI that streams the agent's tool calls and final answer.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- API keys:
  - `GOOGLE_API_KEY` — for Gemini (`gemini-3.1-pro-preview`)
  - `TAVILY_API_KEY` — for Tavily Search

## Setup

1. Open a terminal in this directory:

   ```powershell
   cd 10-langchain/notebooklm
   ```

2. Install dependencies (creates a local `.venv`):

   ```powershell
   uv sync
   ```

   Or with pip:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e .
   ```

3. Create a `.env` file in this directory with your keys:

   ```dotenv
   GOOGLE_API_KEY=your-google-key
   TAVILY_API_KEY=your-tavily-key
   ```

## Run

Launch the Streamlit app (it imports `stream_query` from `agent.py`):

```powershell
uv run streamlit run app.py
```

Or, if you activated the venv with pip:

```powershell
streamlit run app.py
```

Streamlit will open the chat UI at <http://localhost:8501>. To use a different port:

```powershell
uv run streamlit run app.py --server.port 8502
```

## Usage

Type a question in the chat input. The agent will:

1. Rewrite the question into a search query.
2. Call Tavily Search (visible in the "Search trace" expander).
3. Stream a final answer with source URLs.

Use the **Clear chat** button in the sidebar to reset the conversation.

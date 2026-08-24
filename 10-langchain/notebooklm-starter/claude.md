# CLAUDE.md - Development Guide & Instructions

## Core Principles (Token Efficiency)
- **Concise & Direct:** Provide only direct answers and exact code snippets. Do not provide background explanations, unsolicited theory, or verbose summaries.
- **Strict Scope:** Focus strictly on the single active development step. Do not analyze, verify, or scan files outside the immediate scope.
- **No Heavy Operations:** 
  - Do NOT run full test suites, regression tests, recovery checks, or extensive server validations.
  - Do NOT trigger autonomous deep scans or file indexing unless explicitly instructed.
- **Step-by-Step Progress:** Follow the project roadmap sequentially as defined in `mcp.json` / task specifications. Do not jump ahead or run unprompted whole-project audits.

---

## Development Roadmap (Mini-NotebookLM)


### MCP & API Integration
- Target: Expose functionality via MCP server / lightweight API.
- Implementation: Define tools/resources according to `mcp.json`, register handler functions.

---

## Operational Commands (Execute Only When Requested)
- **Run minimal unit check (single file):** `pytest tests/test_<target>.py`
- **Start MCP server:** `python mcp_server.py` or `uvicorn app:app --reload`
- **Clean context:** Use `/clear` or `/compact` after completing each phase.

import json
import uuid

import httpx
import streamlit as st
from agent2 import stream_query

st.set_page_config(page_title="Web Research Agent", page_icon=":material/search:", layout="centered")
st.title("Web Research Agent")
st.caption("Powered by Gemini + Firecrawl")


# Maps each Firecrawl tool to an (icon, friendly label) so the trace makes it
# obvious which tool the agent actually invoked.
TOOL_META = {
    "firecrawl_search": ("🔍", "Search"),
    "firecrawl_scrape": ("📄", "Scrape"),
    "firecrawl_crawl": ("🕸️", "Crawl"),
}


def _tool_meta(name):
    return TOOL_META.get(name or "", ("🛠️", name or "Tool"))


def _render_tool_call(event, number):
    name = event.get("name") or ""
    icon, label = _tool_meta(name)
    st.markdown(f"**{icon} {label}** · call #{number} · `{name}`")
    tool_input = event.get("query") or event.get("url") or ""
    st.code(tool_input or "(no input)", language="text")


def _render_crawl_result(result, result_number):
    url = result.get("url") or ""
    raw = result.get("markdown") or result.get("raw_content") or result.get("content") or ""
    st.markdown(f"{result_number}. [{url}]({url})" if url else f"{result_number}. (no url)")
    if raw:
        preview = raw if len(raw) <= 1000 else raw[:1000] + "..."
        with st.expander("Crawled content", expanded=False):
            st.markdown(preview)


def _render_search_result(result, result_number):
    title = result.get("title") or "Untitled result"
    url = result.get("url") or ""
    content = result.get("content") or result.get("snippet") or result.get("description") or ""
    st.markdown(f"{result_number}. [{title}]({url})" if url else f"{result_number}. {title}")
    if content:
        st.caption(content)


def _render_scrape_result(content):
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    if not text:
        st.caption("(no content)")
        return
    preview = text if len(text) <= 1000 else text[:1000] + "..."
    with st.expander("Scraped content", expanded=False):
        st.markdown(preview)


def _render_tool_result(event, number):
    name = event.get("name") or ""
    icon, label = _tool_meta(name)
    st.markdown(f"**{icon} {label} result** · #{number} · `{name}`")
    content = event.get("content")
    results = event.get("results") or []

    if name == "firecrawl_scrape":
        _render_scrape_result(content)
    elif name == "firecrawl_crawl":
        if results:
            for result_number, result in enumerate(results, start=1):
                _render_crawl_result(result, result_number)
        else:
            st.json(content)
    elif name == "firecrawl_search":
        if results:
            for result_number, result in enumerate(results, start=1):
                _render_search_result(result, result_number)
        else:
            st.json(content)
    else:
        st.json(content)

    st.markdown("Raw tool payload returned to the agent:")
    st.code(
        json.dumps(content, indent=2, ensure_ascii=False, default=str),
        language="json",
    )


def render_usage(usage):
    if not usage:
        return

    if usage.get("summarized"):
        st.success(
            "🧠 Summarization middleware fired — older messages were compressed "
            "into a summary that now stands in for the earlier history."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Current context", f"{usage.get('context_input', 0):,} tok")
    col2.metric("Total (all calls)", f"{usage.get('billed_total', 0):,} tok")
    col3.metric("Msgs in context", usage.get("message_count", 0))

    per_message = usage.get("per_message") or []
    if per_message:
        with st.expander(f"Per-message tokens · {len(per_message)} model calls", expanded=False):
            st.dataframe(
                [
                    {"msg #": p["index"], "input": p["input"],
                     "output": p["output"], "total": p["total"]}
                    for p in per_message
                ],
                use_container_width=True,
                hide_index=True,
            )


def render_search_trace(events):
    if not events:
        return

    with st.expander("Tool trace: what the agent sent to Firecrawl and got back", expanded=True):
        search_count = 0
        result_count = 0
        for event in events:
            if event["type"] == "tool_call":
                search_count += 1
                _render_tool_call(event, search_count)
            else:
                result_count += 1
                _render_tool_result(event, result_count)


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # A stable id per conversation: reused for every message in this session
    # (so the checkpointer keeps memory) and regenerated when the chat is cleared.
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())


def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_search_trace(msg.get("search_events", []))
                render_usage(msg.get("usage"))


def handle_user_prompt(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        trace_area = st.empty()
        response_area = st.empty()
        final_text = ""
        search_events = []
        usage = None

        try:
            for kind, value in stream_query(prompt, st.session_state.conversation_id):
                if kind == "tool_call":
                    search_events.append({"type": "tool_call", **value})
                    icon, label = _tool_meta(value.get("name"))
                    target = value.get("query") or value.get("url") or ""
                    status.info(f"{icon} {label} · {target}")
                elif kind == "tool_result":
                    search_events.append({"type": "tool_result", **value})
                elif kind == "answer":
                    lines = [value["answer"]]
                    if value.get("sources_used"):
                        lines.append("\n**Sources:**")
                        lines += [f"- {url}" for url in value["sources_used"]]
                    if not value.get("answered_from_sources", True):
                        lines.append("\n_Could not be fully answered from the selected sources._")
                    final_text = "\n".join(lines)
                    response_area.markdown(final_text)
                    continue
                elif kind == "usage":
                    usage = value
                    continue
                else:
                    final_text = value
                    response_area.markdown(final_text)
                    continue
                with trace_area.container():
                    render_search_trace(search_events)
        except httpx.ConnectError as exc:
            final_text = (
                "The model/search request could not connect. "
                "Check proxy/VPN/NetFree connectivity and try again.\n\n"
                f"Connection error: `{exc}`"
            )
            response_area.error(final_text)
        except Exception as exc:
            final_text = f"Request failed: `{type(exc).__name__}: {exc}`"
            response_area.error(final_text)
        finally:
            status.empty()

        render_usage(usage)

        st.session_state.messages.append(
            {"role": "assistant", "content": final_text,
             "search_events": search_events, "usage": usage}
        )


def render_sidebar():
    if not st.session_state.messages:
        return
    with st.sidebar:
        st.write(f"Messages: {len(st.session_state.messages)}")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()


init_state()
render_chat_history()
if prompt := st.chat_input("Ask me anything..."):
    handle_user_prompt(prompt)
render_sidebar()

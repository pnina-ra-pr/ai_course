import json

def _parse_tool_content(content):
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def _extract_search_results(parsed_content):
    if isinstance(parsed_content, dict):
        results = parsed_content.get("results")
        if isinstance(results, list):
            return results
    if isinstance(parsed_content, list):
        return parsed_content
    return []


def _message_text(content):
    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                text_blocks.append(block["text"])
        return "\n".join(text_blocks)
    if isinstance(content, str):
        return content
    return ""


def format_message(message):
    """Format agent activity.

    Event shapes:
    - ("tool_call", {"query": str, "name": str})
    - ("tool_result", {"name": str, "content": object, "results": list})
    - ("text", str)
    """
    seen_tool_calls = set()
    seen_tool_results = set()

    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            tc_id = tc.get("id", "")
            if tc_id not in seen_tool_calls:
                seen_tool_calls.add(tc_id)
                args = tc.get("args", {}) or {}
                return "tool_call", {
                    "name": tc.get("name", ""),
                    "query": args.get("query", ""),
                    "url": args.get("url", ""),
                }

    message_type = getattr(message, "type", "")
    if message_type == "tool":
        tool_result_id = getattr(message, "tool_call_id", None) or getattr(message, "id", None)
        if tool_result_id not in seen_tool_results:
            seen_tool_results.add(tool_result_id)
            parsed_content = _parse_tool_content(message.content)
            return "tool_result", {
                "name": getattr(message, "name", ""),
                "content": parsed_content,
                "results": _extract_search_results(parsed_content),
            }

    content = message.content
    text = _message_text(content)
    if text:
        return "text", text


# SummarizationMiddleware replaces the compressed history with a HumanMessage
# whose content starts with this exact marker (see the middleware source).
SUMMARY_MARKER = "Here is a summary of the conversation to date:"


def conversation_usage(messages):
    """Token usage for the current conversation state.

    Only model-generated (AI) messages carry `usage_metadata`; human/tool
    messages are counted implicitly inside the next model call's input tokens.
    Also reports whether SummarizationMiddleware has compressed the history,
    detected via the summary message it injects.
    """
    per_message = []
    billed_total = 0
    context_input = 0
    summarized = False

    for idx, message in enumerate(messages):
        text = _message_text(getattr(message, "content", ""))
        if getattr(message, "type", "") == "human" and text.startswith(SUMMARY_MARKER):
            summarized = True

        usage = getattr(message, "usage_metadata", None)
        if usage:
            in_t = usage.get("input_tokens", 0) or 0
            out_t = usage.get("output_tokens", 0) or 0
            total = usage.get("total_tokens", in_t + out_t) or (in_t + out_t)
            per_message.append({"index": idx, "input": in_t, "output": out_t, "total": total})
            billed_total += total
            context_input = in_t  # last model call reflects the current context size

    return {
        "per_message": per_message,
        "billed_total": billed_total,
        "context_input": context_input,
        "message_count": len(messages),
        "summarized": summarized,
    }
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROLE_STYLES = {
    "HumanMessage": ("[bold blue]User[/bold blue]",      "blue"),
    "AIMessage":    ("[bold green]Agent[/bold green]",   "green"),
    "ToolMessage":  ("[bold yellow]Tool[/bold yellow]",  "yellow"),
}

console = Console()


def extract_content(message):
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, list):
        return "\n".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    return str(content)


def _truncate(s: str, n: int) -> str:
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"\n[dim]… (+{len(s) - n} chars truncated)[/dim]"


def _format_args(args) -> str:
    try:
        return json.dumps(args, ensure_ascii=False, indent=2)
    except Exception:
        return str(args)


def _render_message(msg) -> tuple[str, str, str] | None:
    role = type(msg).__name__
    title, border = ROLE_STYLES.get(role, (role, "white"))

    if role == "AIMessage":
        text = extract_content(msg).strip()
        tool_calls = getattr(msg, "tool_calls", None) or []
        parts = []
        if text:
            parts.append(text)
        if tool_calls:
            parts.append("[dim]→ calling tools:[/dim]")
            for tc in tool_calls:
                name = tc.get("name", "?")
                args = _truncate(_format_args(tc.get("args", {})), 400)
                parts.append(f"  [bold]{name}[/bold]({args})")
        if not parts:
            return None
        return "\n".join(parts), title, border

    if role == "ToolMessage":
        tool_name = getattr(msg, "name", "tool")
        body = _truncate(extract_content(msg), 800)
        return f"[dim]← {tool_name}[/dim]\n{body}", title, border

    return extract_content(msg), title, border


def print_messages(messages):
    for msg in messages:
        rendered = _render_message(msg)
        if rendered is None:
            continue
        body, title, border = rendered
        console.print(Panel(body, title=title, border_style=border))


USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def empty_usage() -> dict:
    return {k: 0 for k in USAGE_KEYS}


def report_usage(messages, session: dict) -> None:
    """Sum token usage from new messages, add to session totals, print a table."""
    turn = empty_usage()
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue
        for key in USAGE_KEYS:
            turn[key] += usage.get(key, 0) or 0
    for key in USAGE_KEYS:
        session[key] += turn[key]

    table = Table(title="Tokens", show_header=True, header_style="bold magenta", expand=False)
    table.add_column("Scope", style="dim")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Total", justify="right")
    table.add_row("This turn", f"{turn['input_tokens']:,}", f"{turn['output_tokens']:,}", f"{turn['total_tokens']:,}")
    table.add_row("Session",   f"{session['input_tokens']:,}", f"{session['output_tokens']:,}", f"{session['total_tokens']:,}")
    console.print(table)

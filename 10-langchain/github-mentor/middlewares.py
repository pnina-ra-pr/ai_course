from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

clear_tool_outputs = ContextEditingMiddleware(
    edits=[
        ClearToolUsesEdit(
            trigger=50000,
            keep=2,
            clear_tool_inputs=False,
            exclude_tools=["get_file_contents"],
            placeholder="[cleared]"
        )
    ]
)
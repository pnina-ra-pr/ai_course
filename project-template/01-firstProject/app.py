import gradio as gr
from llm_service import get_command

def chat(user_input):
    return get_command(user_input)

demo = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="CLI Agent"
)

demo.launch()

from openai import OpenAI
from prompt import SYSTEM_PROMPT

from netfree_unstrict_ssl import unstrict_ssl
unstrict_ssl()

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI()


def get_command(user_input):

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    )

    return response.output_text
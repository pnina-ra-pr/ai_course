from netfree_unstrict_ssl import unstrict_ssl

unstrict_ssl()

from functools import lru_cache

from langchain.chat_models import init_chat_model

MODELS = {
    "planning": "anthropic:claude-haiku-4-5-20251001",  # plan_talk
    "writing": "anthropic:claude-sonnet-5",  # write_talk, revise
    "critique": "anthropic:claude-haiku-4-5-20251001",  # critique
}


@lru_cache
def get_model(role: str):
    return init_chat_model(MODELS[role])

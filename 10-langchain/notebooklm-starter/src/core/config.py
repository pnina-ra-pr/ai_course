"""Shared configuration, read from the environment with sane defaults."""

from __future__ import annotations

import os

MODEL = os.getenv("NOTEBOOKLM_MODEL", "anthropic:claude-sonnet-4-6")

# How much source text an artifact generator is allowed to see in one prompt.
MAX_ARTIFACT_CHARS = int(os.getenv("NOTEBOOKLM_MAX_ARTIFACT_CHARS", "60000"))

"""Studio artifact generators — standalone, stateless agents with structured output.

Each artifact is its own agent: no memory, no tools, one shot. Unlike the chat agent it
does not *search* the notebook, it *covers* it, so every active source is handed to it
verbatim in the prompt.

The point of the stage is ``response_format``: the agent must answer as a Pydantic model
instead of free text, so the result is a validated object the product can rely on. Each
schema knows how to render itself to the markdown the Studio displays.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from core.config import MODEL
from core.store import store

SYSTEM_PROMPT = """You generate artifacts for a notebook of source documents.

Rules:
- Ground every statement in the supplied sources. Never invent facts, figures or names.
- Where a field asks for a source, give the exact name from that source's `name` attribute.
- If the sources do not cover something, leave it out rather than guessing.
- Write in the dominant language of the sources.
"""


# -- summary -------------------------------------------------------------------


class KeyPoint(BaseModel):
    heading: str = Field(description="A label for the point, at most six words.")
    detail: str = Field(description="One or two sentences explaining it.")
    source: str = Field(description="Exact name of the source this point comes from.")


class Summary(BaseModel):
    """A structured brief of the notebook as a whole."""

    title: str = Field(description="A title for the notebook as a whole.")
    overview: str = Field(description="Two to four sentences on what the sources cover.")
    key_points: list[KeyPoint] = Field(
        description="The four to seven most important points across all sources."
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions the sources raise but do not answer. Empty if none.",
    )

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", self.overview, "", "## Key points", ""]
        for point in self.key_points:
            lines.append(f"- **{point.heading}** — {point.detail}  \n  _{point.source}_")
        if self.open_questions:
            lines += ["", "## Open questions", ""]
            lines += [f"- {question}" for question in self.open_questions]
        return "\n".join(lines)


# -- faq -----------------------------------------------------------------------


class QAItem(BaseModel):
    question: str = Field(description="A question a reader of these sources would ask.")
    answer: str = Field(description="Two to four sentences, answered from the sources only.")
    source: str = Field(description="Exact name of the source the answer comes from.")


class FAQ(BaseModel):
    """The questions the notebook can actually answer."""

    title: str = Field(description="A title for the FAQ.")
    items: list[QAItem] = Field(
        description="Five to eight questions, ordered from most basic to most specific."
    )

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for item in self.items:
            lines += [f"**{item.question}**", "", item.answer, "", f"_{item.source}_", ""]
        return "\n".join(lines).rstrip()


# -- infographic ---------------------------------------------------------------


class Stat(BaseModel):
    value: str = Field(description="The figure itself, e.g. '42%', '$1.2M', '3x'.")
    label: str = Field(description="What the figure measures, at most five words.")
    context: str | None = Field(
        default=None, description="One short clause of context, if it helps. Otherwise omit."
    )


class Section(BaseModel):
    heading: str = Field(description="A theme found in the sources, at most six words.")
    bullets: list[str] = Field(description="Two to four bullets, one line each.")


class Infographic(BaseModel):
    """A poster-style breakdown: headline figures plus themed sections."""

    title: str = Field(description="A short, punchy title.")
    subtitle: str = Field(description="One line framing the topic.")
    stats: list[Stat] = Field(
        description="Three to five headline figures, taken verbatim from the sources."
    )
    sections: list[Section] = Field(description="Three or four themed sections.")
    takeaway: str = Field(description="The one sentence a reader should leave with.")

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", f"_{self.subtitle}_", "", "## By the numbers", ""]
        for stat in self.stats:
            context = f" — {stat.context}" if stat.context else ""
            lines.append(f"- **{stat.value}** · {stat.label}{context}")
        for section in self.sections:
            lines += ["", f"## {section.heading}", ""]
            lines += [f"- {bullet}" for bullet in section.bullets]
        lines += ["", "---", "", f"**{self.takeaway}**"]
        return "\n".join(lines)


# -- the generators ------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """What makes one artifact different from another: its shape and its instruction."""

    schema: type[BaseModel]
    task: str


SPECS: dict[str, Spec] = {
    "summary": Spec(
        schema=Summary,
        task="Write the summary of this notebook.",
    ),
    "faq": Spec(
        schema=FAQ,
        task="Write the FAQ for this notebook — the questions these sources answer.",
    ),
    "infographic": Spec(
        schema=Infographic,
        task=(
            "Design an infographic for this notebook. Prefer concrete figures from the "
            "sources over vague claims; if the sources contain few numbers, use the most "
            "quotable specifics instead."
        ),
    ),
}


def generate(kind: str) -> tuple[str, str]:
    """Run the artifact agent for ``kind``; return its ``(title, markdown)``.

    Raises ``KeyError`` for a kind with no generator.
    """
    spec = SPECS[kind]

    agent = create_agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        response_format=spec.schema,
    )

    prompt = f"{spec.task}\n\nSources:\n\n{store.active_content()}"
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    artifact = result["structured_response"]
    return artifact.title, artifact.to_markdown()

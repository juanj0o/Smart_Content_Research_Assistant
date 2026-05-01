"""Reporter node — produces the final markdown report. Model tier: PREMIUM."""

import json

from langchain_core.prompts import ChatPromptTemplate

from config import get_model
from cost_tracker import make_cost_entry
from graph_state import ResearchState
from llm_factory import get_llm


# The reporter receives a numbered source index so it can write inline
# citations like [1] [2] in the prose. We build that index in Python
# (deterministically) and pass it in, then append a clean References
# section after the LLM responds — never trust an LLM to format URLs.
SYSTEM_PROMPT = """You are the Reporter Agent. You take curated research
content and write a polished, professional markdown research report.

Each section will include a list of numbered sources for that subtopic.
Weave inline citations into the prose using the format [N] wherever the
claim comes from a source (e.g. "Recent breakthroughs have shown [1][3]...").

Follow this exact structure:

# Research Report: {{topic}}

## Executive Summary
<concise high-level summary, 3-5 sentences — no citations needed here>

## Introduction
<framing of the topic and why it matters>

## <One section per subtopic, using the subtopic name as heading>
<detailed writing built from deep_analysis and key_insights; include [N]
inline citations where appropriate>

## Cross-cutting Insights
<themes that emerged across multiple sections>

## Conclusions
<synthesized takeaways and forward-looking notes>

IMPORTANT:
- Do NOT write a References section — that will be appended automatically.
- Do NOT include a trailing metadata line.
- Only use citation numbers that appear in the sources list you are given.
- Output only the markdown, no code fences around the whole document."""


def _build_source_index(curated: dict) -> tuple[dict[str, list[str]], list[tuple[int, str, str]]]:
    """
    Build two structures from curated_sections:

    per_section  → {subtopic: [url, ...]}  (ordered, deduplicated per section)
    global_index → [(global_n, subtopic, url), ...]

    The global_index is what goes into the ## References block.
    Numbers are assigned in section order, so [1] always means the same URL
    throughout the document.
    """
    per_section: dict[str, list[str]] = {}
    global_index: list[tuple[int, str, str]] = []
    seen_urls: set[str] = set()
    counter = 1

    for section in curated.get("curated_sections", []):
        subtopic = section.get("subtopic", "")
        sources  = section.get("sources", [])
        local_urls: list[str] = []

        for url in sources:
            url = url.strip()
            if not url:
                continue
            if url not in seen_urls:
                seen_urls.add(url)
                global_index.append((counter, subtopic, url))
                counter += 1
            # local list keeps only urls for this section (for the prompt)
            local_urls.append(url)

        per_section[subtopic] = local_urls

    return per_section, global_index


def _build_references_md(global_index: list[tuple[int, str, str]]) -> str:
    """Render the References section deterministically — no LLM involved."""
    if not global_index:
        return ""

    lines = ["\n## References\n"]
    current_subtopic = None

    for n, subtopic, url in global_index:
        if subtopic != current_subtopic:
            lines.append(f"\n**{subtopic}**\n")
            current_subtopic = subtopic
        lines.append(f"[{n}] {url}")

    return "\n".join(lines)


def _annotate_payload(curated: dict, per_section: dict[str, list[str]]) -> dict:
    """
    Return a copy of curated where each section has a numbered source list
    in the format the LLM prompt expects: ["[1] https://...", "[2] ..."]
    This makes inline citation easier — the LLM can see "this fact → [2]".
    """
    # Build a global url→number map first
    url_to_n: dict[str, int] = {}
    n = 1
    for section in curated.get("curated_sections", []):
        for url in section.get("sources", []):
            url = url.strip()
            if url and url not in url_to_n:
                url_to_n[url] = n
                n += 1

    annotated_sections = []
    for section in curated.get("curated_sections", []):
        s = dict(section)
        numbered = [
            f"[{url_to_n[u.strip()]}] {u.strip()}"
            for u in s.get("sources", [])
            if u.strip() in url_to_n
        ]
        s["sources"] = numbered
        annotated_sections.append(s)

    return {**curated, "curated_sections": annotated_sections}


def reporter_node(state: ResearchState) -> dict:
    """Reads curated_content and topic; writes the final report with references."""
    model   = get_model("premium")
    llm     = get_llm(model, temperature=0)
    curated = state.get("curated_content", {})

    # Build the source index before calling the LLM
    per_section, global_index = _build_source_index(curated)
    annotated_curated         = _annotate_payload(curated, per_section)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",
         "Write the final markdown research report using the following "
         "curated content. Each section includes numbered sources — use "
         "them for inline citations.\n\n{payload}"),
    ])

    payload = json.dumps(
        {"topic": state["topic"], "curated": annotated_curated},
        indent=2,
    )

    chain    = prompt | llm
    response = chain.invoke({"payload": payload})

    text = response.content if isinstance(response.content, str) else str(response.content)

    # Append the References block — built in Python, not by the LLM,
    # so URLs are always accurate and consistently formatted.
    references_md = _build_references_md(global_index)
    final_text    = text.strip() + references_md

    cost_entry = make_cost_entry("Reporter", model, getattr(response, "usage_metadata", None))
    return {"final_report": final_text, "cost_log": [cost_entry]}

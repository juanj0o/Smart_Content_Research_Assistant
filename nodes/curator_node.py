"""Curator node — deeper analysis of approved subtopics. Model tier: SMART."""

import json

from langchain_core.prompts import ChatPromptTemplate

from config import get_model
from cost_tracker import make_cost_entry
from graph_state import ResearchState
from json_utils import parse_json_robust
from llm_factory import get_llm


# The curator now includes `sources` in both input and output so they
# travel all the way through to the reporter without being dropped.
SYSTEM_PROMPT = """You are the Curator Agent. You take a set of approved
subtopics (each with real web sources) and produce a deeper analytical layer:
strong paragraphs of analysis, distilled insights, a relevance score (0-10),
cross-links between subtopics, and the sources that back up the analysis.

CRITICAL OUTPUT RULES — follow these exactly:
1. Output ONLY a single valid JSON object. No prose before or after.
2. Do NOT wrap the JSON in markdown fences (no ```json, no ```).
3. Do NOT add // comments inside the JSON.
4. Do NOT include trailing commas before }} or ].
5. Use only straight ASCII quotes ("), never curly quotes (“ ”).
6. If a subtopic has a sparse summary or empty key_points, still produce a
   complete section for it — write the deep_analysis from your own knowledge
   of the subtopic name.
7. Always copy the input "sources" list into the output section verbatim.
   Do NOT invent new URLs. If sources is empty, output sources: [].

Required schema:

{{
  "curated_sections": [
    {{
      "subtopic": "<name>",
      "deep_analysis": "<a full analytical paragraph, 4-7 sentences>",
      "key_insights": ["<insight>", "..."],
      "relevance_score": 8.5,
      "connections_to_other_subtopics": ["<name or note>", "..."],
      "sources": ["<url>", "..."]
    }}
  ],
  "overall_synthesis": "<a paragraph weaving the subtopics together>"
}}"""


_FALLBACK = {"curated_sections": [], "overall_synthesis": ""}


def _build_local_fallback(approved: list[dict]) -> dict:
    """
    Final safety net: if the LLM JSON cannot be repaired AT ALL, build a
    minimal-but-valid curated_content dict from the approved subtopics so
    the reporter still has something to write about (and sources survive).
    """
    sections = []
    for s in approved:
        sections.append({
            "subtopic":                       s.get("subtopic", ""),
            "deep_analysis":                  s.get("summary", "") or
                                              f"Analysis pending for {s.get('subtopic','')}.",
            "key_insights":                   s.get("key_points", []),
            "relevance_score":                7.0,
            "connections_to_other_subtopics": [],
            "sources":                        s.get("sources", []),
        })
    return {
        "curated_sections":  sections,
        "overall_synthesis": "Synthesis unavailable — curator output could not be parsed.",
    }


def curator_node(state: ResearchState) -> dict:
    """Reads approved_subtopics from state, returns deeper analysis with sources."""
    model = get_model("smart")
    llm = get_llm(model, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",
         "Analyze the following approved subtopics and produce the JSON "
         "curation as specified.\n\n{payload}"),
    ])

    payload = json.dumps({
        "topic": state["topic"],
        "approved_subtopics": [
            {
                "subtopic":   s.get("subtopic", ""),
                "summary":    s.get("summary", ""),
                "key_points": s.get("key_points", []),
                "sources":    s.get("sources", []),   # ← pass real URLs through
            }
            for s in state.get("approved_subtopics", [])
        ],
    }, indent=2)

    chain = prompt | llm
    response = chain.invoke({"payload": payload})

    text = response.content if isinstance(response.content, str) else str(response.content)
    parsed = parse_json_robust(text, _FALLBACK, label="Curator")

    # If the robust parser still couldn't extract sections, fall back to a
    # locally-constructed dict so the reporter never sees empty data.
    approved = state.get("approved_subtopics", [])
    if not parsed.get("curated_sections"):
        print("  ↪︎ Building local curator fallback from approved subtopics.")
        parsed = _build_local_fallback(approved)

    # Safety net: if the LLM forgot to copy sources, restore them from
    # the approved_subtopics that are already in state.
    source_map = {
        s.get("subtopic", ""): s.get("sources", [])
        for s in approved
    }
    for section in parsed.get("curated_sections", []):
        if not section.get("sources"):
            section["sources"] = source_map.get(section.get("subtopic", ""), [])

    cost_entry = make_cost_entry("Curator", model, getattr(response, "usage_metadata", None))
    return {"curated_content": parsed, "cost_log": [cost_entry]}

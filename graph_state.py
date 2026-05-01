"""
The graph's shared state — the single most important concept in LangGraph.

─────────────────────────────────────────────────────────────────────────────
WHAT IS A TypedDict?
─────────────────────────────────────────────────────────────────────────────
A TypedDict is NOT a class with methods. It is a plain `dict` at runtime,
with type annotations for each key. LangGraph picks it because a plain dict
is trivial to serialize (JSON-friendly), trivial to checkpoint, and gives
us field-level type hints without the ceremony of a full class.

    state = {"topic": "AI", "raw_findings": [...], ...}
    # Still a dict! Access with state["topic"], not state.topic.

─────────────────────────────────────────────────────────────────────────────
WHAT DOES `Annotated[list, operator.add]` DO? (REDUCERS)
─────────────────────────────────────────────────────────────────────────────
Every time a node returns a partial update like `{"raw_findings": [...]}`,
LangGraph has to MERGE that update into the existing state. The default
merge strategy is "replace" — the new value overwrites the old.

A reducer lets us override that. `operator.add` on a list means:
    new_state["raw_findings"] = old_state["raw_findings"] + update

That's critical here for `cost_log`: every agent node appends one entry,
and we want those entries accumulated across nodes, not overwritten.

─────────────────────────────────────────────────────────────────────────────
WHY NOT A PLAIN DICT OR A DATACLASS?
─────────────────────────────────────────────────────────────────────────────
1. LangGraph uses this type to VALIDATE updates — nodes returning unknown
   keys would silently drop data otherwise.
2. The checkpointer (MemorySaver / PostgresSaver / RedisSaver) serializes
   state after every node. A TypedDict is already dict-shaped, so no custom
   encoder is needed.
3. Reducers only work via `Annotated[...]`, which requires type annotations
   — hence TypedDict, not a bare dict.
"""

import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict):
    # The research topic entered by the user. Set once at the start,
    # so no reducer — default "replace" behavior is fine.
    topic: str

    # Raw findings produced by the Investigator node.
    # Reducer: operator.add → LangGraph APPENDS to the list instead of
    # replacing it. If multiple nodes contributed findings, they'd stack.
    raw_findings: Annotated[list[dict], operator.add]

    # Subtopics approved by the human. Set once by the human_validation
    # node, so "replace" semantics are what we want → no reducer.
    approved_subtopics: list[dict]

    # Deep analysis from the Curator node. Single write → no reducer.
    curated_content: dict

    # Final markdown report string. Single write → no reducer.
    final_report: str

    # Raw web search results fetched by the Investigator before calling the
    # LLM. Stored here so the curator / reporter could also reference them
    # later if needed. operator.add → accumulated, never replaced.
    # Each entry: {"query": str, "results": [{"title", "url", "content"}]}
    search_results: Annotated[list[dict], operator.add]

    # Cost tracking entries, one per agent call. Each node appends one
    # entry, so we use operator.add to accumulate across the whole run.
    cost_log: Annotated[list[dict], operator.add]

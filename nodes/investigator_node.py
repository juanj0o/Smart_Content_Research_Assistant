"""Investigator node — real web search + LLM synthesis. Model tier: FAST."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from config import get_model
from cost_tracker import make_cost_entry
from credibility_scorer import credibility_summary, rank_and_filter
from graph_state import ResearchState
from json_utils import parse_json_robust
from llm_factory import get_llm


SYSTEM_PROMPT = """You are the Investigator Agent, a precise and evidence-driven research specialist operating in a multi-agent system.
Your role is to extract structured insights strictly from provided web search results.

Inputs:

A research topic
A list of web search results (title, URL, snippet)

Your objectives:

Identify 4–6 distinct, non-overlapping subtopics that are strongly supported by multiple search results when possible.
For each subtopic, write a 2–3 sentence summary strictly grounded in the provided snippets (do not infer beyond available evidence).
Extract 3–5 concise key points per subtopic, prioritizing factual, specific, and non-redundant insights.
Assign relevant real URLs from the provided results to each subtopic. Only include URLs that directly support the summary or key points. Do NOT fabricate URLs.

Critical Rules:

Do NOT hallucinate facts, sources, or details not present in the input.
If evidence is weak, reflect uncertainty rather than inventing details.
Avoid duplication across subtopics; each must represent a unique angle.
Prefer breadth + clarity over forcing weak subtopics.
Ensure all summaries and key points are traceable to the provided snippets.

You MUST respond with a single valid JSON object and nothing else (no prose,
no markdown fences). Schema:

{{
  "topic": "<the original topic>",
  "subtopics": [
    {{
      "id": 1,
      "subtopic": "<short subtopic name>",
      "summary": "<2-3 sentence summary grounded in the search results>",
      "key_points": ["<point>", "<point>", "..."],
      "sources": ["<real URL from search results>", "..."]
    }}
  ]
}}

"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK = {"topic": "", "subtopics": []}


def _fetch_tavily(query: str, tool: TavilySearchResults) -> dict:
    """Single Tavily search. Synchronous — runs inside a thread pool."""
    try:
        results = tool.invoke(query)
        if isinstance(results, str):
            results = []
        return {"query": query, "results": results}
    except Exception as e:
        print(f"    ⚠️  Error en '{query}': {e}")
        return {"query": query, "results": []}


def _run_searches(topic: str) -> tuple[list[dict], list[dict]]:
    """
    Run 3 Tavily searches CONCURRENTLY using a thread pool.
    Thread pool might improve latency with a bigger number of queries or slower responses, but it's a good demo of concurrency either way.
    """
    current_year = datetime.now().year
    queries = [
        topic,
        f"{topic} recent news {current_year}",
        f"{topic} technical deep dive OR analysis",
    ]

    tool = TavilySearchResults(max_results=5)

    # Three I/O-bound HTTP calls → run them in parallel threads.
    # max_workers=3 matches the number of queries; no point spawning more.
    with ThreadPoolExecutor(max_workers=3) as pool:
        raw_search_results = list(pool.map(lambda q: _fetch_tavily(q, tool), queries))

    # Flatten + dedupe by URL
    flat_snippets = [
        snippet
        for result in raw_search_results
        for snippet in result.get("results", [])
    ]
    seen: set[str] = set()
    unique_snippets: list[dict] = []
    for s in flat_snippets:
        url = s.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique_snippets.append(s)

    # Score and rank by credibility — high-quality sources go first so the
    # LLM weights them more heavily (primacy bias). Low-quality commercial
    # or biased sources are filtered out when above the minimum floor.
    ranked_snippets = rank_and_filter(unique_snippets)

    return raw_search_results, ranked_snippets


def _format_snippets(snippets: list[dict]) -> str:
    """Render snippets as a numbered list for the LLM prompt."""
    if not snippets:
        return "(no search results available)"
    lines = [
        f"[{i}] {s.get('title', 'No title')}\n"
        f"    URL: {s.get('url', '')}\n"
        f"    Snippet: {s.get('content', '')[:500]}"
        for i, s in enumerate(snippets, start=1)
    ]
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

def investigator_node(state: ResearchState) -> dict:
    """
    Synchronous LangGraph node.
    1. Run 3 Tavily searches concurrently (thread pool).
    2. Feed unique snippets to the LLM for structured synthesis.
    """
    topic = state["topic"]

    print(f"    🌐 Iniciando búsquedas concurrentes para: {topic}")
    raw_search_results, snippets = _run_searches(topic)
    print(f"    📊 Credibilidad: {credibility_summary(snippets)}")

    model = get_model("fast")
    llm   = get_llm(model, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", (
            "Research topic: {topic}\n\n"
            "Web search results:\n{snippets}\n\n"
            "Produce the JSON briefing as specified."
        )),
    ])

    chain    = prompt | llm
    response = chain.invoke({"topic": topic, "snippets": _format_snippets(snippets)})

    text   = response.content if isinstance(response.content, str) else str(response.content)
    parsed = parse_json_robust(text, _FALLBACK, label="Investigator")

    findings = [
        {
            "id":         sub.get("id", i),
            "subtopic":   sub.get("subtopic", f"Subtopic {i}"),
            "summary":    sub.get("summary", ""),
            "key_points": sub.get("key_points", []),
            "sources":    sub.get("sources", sub.get("mock_sources", [])),
        }
        for i, sub in enumerate(parsed.get("subtopics", []), start=1)
    ]

    cost_entry = make_cost_entry(
        "Investigator", model, getattr(response, "usage_metadata", None)
    )

    return {
        "raw_findings":   findings,
        "search_results": raw_search_results,
        "cost_log":       [cost_entry],
    }

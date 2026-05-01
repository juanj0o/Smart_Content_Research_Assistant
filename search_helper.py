"""
Shared Tavily search helpers.

Both the investigator (multi-query broad sweep) and the human node (single
quick lookup when the user adds a custom subtopic) need to call Tavily.
"""

from langchain_community.tools.tavily_search import TavilySearchResults

from credibility_scorer import rank_and_filter


def quick_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Single Tavily search. Returns a list of {"title", "url", "content"} dicts,
    or an empty list on any failure (network, missing API key, etc.).

    Used by the human node to enrich a user-added subtopic with real sources.
    """
    try:
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke(query)
        if not isinstance(results, list):
            return []
        return rank_and_filter(results)
    except Exception as e:
        print(f"    ⚠️  Quick search failed for '{query}': {e}")
        return []


def synthesize_subtopic_from_results(name: str, results: list[dict]) -> dict:
    """
    Build a subtopic dict from raw Tavily results — no LLM call.

    summary: first ~280 chars of the top snippet (heuristic, but real text)
    key_points: titles of the top 3 results (always real)
    sources: the URLs of all results

    This avoids the "summary empty + sources empty" pathological case that
    used to trip the curator into producing unparseable JSON.
    """
    if not results:
        return {
            "summary": f"User-added subtopic: {name}. No search results available.",
            "key_points": [],
            "sources": [],
        }

    top = results[0]
    snippet = (top.get("content") or "").strip()
    summary = snippet[:280] + ("…" if len(snippet) > 280 else "")
    if not summary:
        summary = f"User-added subtopic: {name}."

    key_points = [r.get("title", "").strip() for r in results[:3] if r.get("title")]
    sources    = [r.get("url",   "").strip() for r in results     if r.get("url")]

    return {
        "summary":    summary,
        "key_points": key_points,
        "sources":    sources,
    }

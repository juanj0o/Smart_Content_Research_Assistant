"""
Source credibility scorer — Level 2 domain-based heuristic.

Approach: score each Tavily result from 0.0 to 1.0 based on domain signals,
then sort high-quality sources first (primacy bias) and filter out clearly
commercial/biased ones.

Why this matters: Tavily returns whatever Google ranks highly, which includes
commercial sites with financial incentives to present biased information.
A site selling EMF protection products has a direct incentive to overstate
5G health risks — it should not carry the same weight as a WHO or PMC source.

Design decisions:
- Conservative filter threshold (0.15): only drops clearly commercial/biased
  results, not borderline ones. Better to keep a questionable source than
  leave the LLM with too little context.
- Minimum results floor: always keep at least MIN_RESULTS snippets regardless
  of score, so the investigator never starts with an empty context.
- Sort by score descending: high-quality sources appear first in the prompt.
  LLMs exhibit primacy bias — they weight earlier context more heavily.
"""

from urllib.parse import urlparse


# ── Tunable constants ─────────────────────────────────────────────────────────
FILTER_THRESHOLD = 0.15   # drop results below this score
MIN_RESULTS      = 5      # always keep at least this many, even if low quality


# ── Domain lists ──────────────────────────────────────────────────────────────

# Domains with strong credibility signals — receiving a score bonus.
# Uses suffix matching so subdomains (e.g. pmc.ncbi.nlm.nih.gov) also match.
HIGH_CREDIBILITY_DOMAINS: set[str] = {
    # Scientific journals & preprints
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "arxiv.org", "biorxiv.org", "medrxiv.org",
    "nature.com", "science.org", "cell.com", "thelancet.com", "nejm.org",
    "bmj.com", "jamanetwork.com", "springer.com", "sciencedirect.com",
    "wiley.com", "tandfonline.com", "ieee.org", "acm.org", "oup.com",
    "cambridge.org", "royalsocietypublishing.org",
    # Health & government
    "who.int", "cdc.gov", "nih.gov", "ecdc.europa.eu", "europarl.europa.eu",
    "nist.gov", "fda.gov", "fcc.gov", "icnirp.org",
    # Trusted news
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "theguardian.com", "ft.com", "economist.com", "wsj.com",
    "bloomberg.com", "nytimes.com",
    # Tech & research
    "ibm.com", "research.ibm.com", "aws.amazon.com", "cloud.google.com",
    "research.google.com", "deepmind.com", "openai.com", "anthropic.com",
    "microsoft.com", "research.microsoft.com",
    # Universities (top institutions — not exhaustive)
    "mit.edu", "stanford.edu", "harvard.edu", "ox.ac.uk", "cam.ac.uk",
    "caltech.edu", "cmu.edu", "berkeley.edu", "columbia.edu",
    # Consulting / standards (authoritative for business / compliance topics)
    "ey.com", "mckinsey.com", "deloitte.com", "pwc.com",
    "isaca.org", "iso.org", "nist.gov",
}

# URL path fragments that signal a commercial / transactional page.
_COMMERCIAL_URL_SIGNALS: tuple[str, ...] = (
    "/shop", "/store", "/buy-", "/product", "/cart",
    "/checkout", "/affiliate", "/promo", "discount",
    "coupon", "ref=", "utm_campaign",
)

# Content phrases typical of product-selling pages.
_COMMERCIAL_CONTENT_SIGNALS: tuple[str, ...] = (
    "buy now", "add to cart", "free shipping", "shop now",
    "our products protect", "sale price", "% off", "order today",
    "money-back guarantee", "as seen on",
)

# Title phrases typical of clickbait / low-quality content.
_CLICKBAIT_SIGNALS: tuple[str, ...] = (
    "you won't believe", "shocking truth", "doctors hate",
    "one weird trick", "secret they don't", "big pharma",
    "mainstream media won't", "wake up",
)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_result(result: dict) -> float:
    """
    Return a credibility score in [0.0, 1.0].

    Baseline 0.5 → adjusted up for trusted signals, down for bias signals.
    """
    url     = (result.get("url")     or "").lower()
    title   = (result.get("title")   or "").lower()
    content = (result.get("content") or "").lower()

    try:
        parsed = urlparse(url)
        # Use removeprefix, not lstrip — lstrip("www.") treats its argument
        # as a character SET and would corrupt domains like "who.int" → "ho.int"
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
    except Exception:
        domain = ""

    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""

    score = 0.5  # neutral baseline

    # ── Positive signals ─────────────────────────────────────────────────
    if tld == "gov":
        score += 0.35
    elif tld == "edu":
        score += 0.30
    elif tld == "org":
        score += 0.08   # not all .org are trustworthy, but generally better

    # High-credibility domain bonus (suffix match for subdomains)
    for hq in HIGH_CREDIBILITY_DOMAINS:
        if domain == hq or domain.endswith("." + hq):
            score += 0.30
            break

    # ── Negative signals ─────────────────────────────────────────────────
    if any(sig in url for sig in _COMMERCIAL_URL_SIGNALS):
        score -= 0.40

    if any(sig in content for sig in _COMMERCIAL_CONTENT_SIGNALS):
        score -= 0.30

    if any(sig in title for sig in _CLICKBAIT_SIGNALS):
        score -= 0.35

    return round(max(0.0, min(1.0, score)), 3)


def rank_and_filter(results: list[dict]) -> list[dict]:
    """
    Score every result, sort by credibility descending, then filter.

    Filtering strategy:
    - Always keep the top MIN_RESULTS results regardless of score.
    - Beyond that, drop anything below FILTER_THRESHOLD.

    This prevents the scorer from leaving the LLM with an empty context
    on niche topics where all sources are somewhat low-quality blogs.
    """
    if not results:
        return []

    scored = [(score_result(r), r) for r in results]
    scored.sort(key=lambda x: x[0], reverse=True)  # best first

    filtered: list[dict] = []
    for i, (score, result) in enumerate(scored):
        keep_anyway = i < MIN_RESULTS          # always keep the top N
        above_threshold = score >= FILTER_THRESHOLD
        if keep_anyway or above_threshold:
            result = dict(result)
            result["_credibility_score"] = score   # attach for logging
            filtered.append(result)

    return filtered


def credibility_summary(results: list[dict]) -> str:
    """
    Return a one-line console summary of source quality distribution.
    Used by the investigator node for transparent logging.
    """
    if not results:
        return "no results"

    scores = [r.get("_credibility_score", score_result(r)) for r in results]
    high   = sum(1 for s in scores if s >= 0.70)
    medium = sum(1 for s in scores if 0.35 <= s < 0.70)
    low    = sum(1 for s in scores if s < 0.35)

    return f"{len(scores)} sources — 🟢 {high} high  🟡 {medium} medium  🔴 {low} low"

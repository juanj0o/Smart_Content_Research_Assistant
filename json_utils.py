"""
Robust JSON parsing helpers shared by all nodes.

The previous per-node `_parse_json` was very fragile against the kinds of
output small models (Llama 3.1 8B, Qwen 2.5 7B) frequently produce:

  - Markdown fences around the JSON
  - Leading prose ("Here is the JSON:") or trailing prose
  - Trailing commas before closing braces / brackets
  - Truncated JSON when max_tokens is reached mid-output
  - Inline `// comments` the model invents
  - Smart quotes instead of straight quotes
  - Stray control characters

`parse_json_robust()` applies a series of escalating repairs so a usable
dict comes out even from messy responses.
"""

import json
import re


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_robust(text: str, fallback: dict, label: str = "JSON") -> dict:
    """
    Try increasingly aggressive strategies to coerce `text` into a dict.
    Returns `fallback` (a copy) only as a last resort, with a console warning
    that names the calling site via `label` for easier debugging.
    """
    if not text or not text.strip():
        print(f"  ⚠️  {label}: empty response. Using fallback.")
        return dict(fallback)

    candidates = _generate_candidates(text)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    print(f"  ⚠️  {label}: returned unparseable JSON. Using fallback.")
    return dict(fallback)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _generate_candidates(text: str) -> list[str]:
    """Yield repair attempts in order of decreasing fidelity to the original."""
    candidates: list[str] = []

    # 1. As-is, just trimmed.
    base = text.strip()
    candidates.append(base)

    # 2. Strip markdown fences.
    fenced = _strip_fences(base)
    if fenced != base:
        candidates.append(fenced)

    # 3. Normalize smart quotes and remove BOM / control noise.
    normalized = _normalize_quotes(fenced)
    if normalized != fenced:
        candidates.append(normalized)

    # 4. Extract the outermost balanced {...} block — drops surrounding prose.
    balanced = _extract_balanced_object(normalized)
    if balanced and balanced != normalized:
        candidates.append(balanced)

    # 5. Strip // line-comments the model may have invented.
    no_comments = _strip_line_comments(balanced or normalized)
    if no_comments not in candidates:
        candidates.append(no_comments)

    # 6. Remove trailing commas before closing braces/brackets.
    no_trailing = _strip_trailing_commas(no_comments)
    if no_trailing not in candidates:
        candidates.append(no_trailing)

    # 7. Auto-close truncated JSON (max_tokens cut-off).
    closed = _autoclose(no_trailing)
    if closed not in candidates:
        candidates.append(closed)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Individual repair primitives
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_quotes(text: str) -> str:
    # Replace smart quotes with straight quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    # Strip BOM
    text = text.lstrip("﻿")
    return text


def _extract_balanced_object(text: str) -> str:
    """
    Walk the string and return the substring from the first `{` to the
    matching `}`, respecting string literals so braces inside quoted
    strings don't throw off the depth counter.
    """
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Truncated → return from start to end so _autoclose can patch it.
    return text[start:]


def _strip_line_comments(text: str) -> str:
    """Remove `// ...` comments outside strings. Crude but good enough."""
    out_lines = []
    for line in text.splitlines():
        # Find // outside any quotes
        in_str = False
        escape = False
        cut = -1
        for i, ch in enumerate(line):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str and ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                cut = i
                break
        out_lines.append(line if cut < 0 else line[:cut].rstrip())
    return "\n".join(out_lines)


def _strip_trailing_commas(text: str) -> str:
    # Remove commas that come right before a closing brace or bracket.
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _autoclose(text: str) -> str:
    """
    Close any unbalanced `{` / `[` left over after truncation. Walks the
    string respecting string literals, then appends matching closers.
    Also closes a dangling open string if needed.
    """
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    closer = ""
    if in_string:
        closer += '"'
    # Drop dangling trailing commas before we close.
    trimmed = text.rstrip().rstrip(",")
    while stack:
        closer += stack.pop()
    return trimmed + closer

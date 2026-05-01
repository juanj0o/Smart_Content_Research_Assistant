"""Human-in-the-loop node — pauses the graph for human review.

This is the most pedagogically important node in the project: it uses
LangGraph's `interrupt()` primitive to PAUSE the graph mid-run, and
returns a `Command` object that both updates state AND routes to the
next node.
"""

import re

from langgraph.types import Command, interrupt

from graph_state import ResearchState
from search_helper import quick_search, synthesize_subtopic_from_results


SEP = "══════════════════════════════════════════════"


# ─────────────────────────────────────────────────────────────────────────
# Console rendering (identical UX to the from-scratch version)
# ─────────────────────────────────────────────────────────────────────────

def _render_findings(topic: str, findings: list[dict]) -> None:
    print()
    print(SEP)
    print("  RESEARCH FINDINGS — Human Validation Required")
    print(SEP)
    print()
    print(f"Topic: {topic}")
    print()
    print("Subtopics found:")
    for item in findings:
        print(f"  [{item['id']}] {item['subtopic']}")
        if item.get("summary"):
            print(f"      Summary: {item['summary']}")
        if item.get("key_points"):
            print(f"      Key points: {', '.join(item['key_points'])}")
        print()


def _print_commands() -> None:
    print("Commands:")
    print("  approve all              → approve everything")
    print("  approve 1,3              → approve specific items")
    print("  reject 2                 → remove an item")
    print('  modify 1 "new name"      → rename a subtopic')
    print('  add "new subtopic"       → add a custom subtopic')
    print("  done                     → finish editing")
    print()
    print('You can chain commands: "approve 1,3 | reject 2 | add \'AI regulation\'"')
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────
# Command parsing (ported from the from-scratch version)
# ─────────────────────────────────────────────────────────────────────────

def _parse_ids(arg: str) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    invalid: list[str] = []
    for chunk in arg.split(","):
        token = chunk.strip()
        if not token:
            continue
        if token.isdigit():
            ids.append(int(token))
        else:
            invalid.append(token)
    return ids, invalid


def _extract_quoted(s: str) -> str | None:
    m = re.search(r'["\']([^"\']+)["\']', s)
    return m.group(1) if m else None


def _apply_command(
    cmd: str,
    pool: dict[int, dict],
    approved: dict[int, dict],
    next_id: list[int],
) -> bool:
    """Mutates pool/approved in place. Returns True on `done`."""
    cmd = cmd.strip()
    if not cmd:
        return False
    lowered = cmd.lower()

    if lowered == "done":
        return True

    if lowered.startswith("approve"):
        rest = cmd[len("approve"):].strip()
        if rest.lower() == "all" or rest == "":
            for id_, item in pool.items():
                approved[id_] = item
            print(f"  ✓ Approved all {len(pool)} subtopics.")
            return False
        ids, invalid = _parse_ids(rest)
        if invalid:
            print(f"  ✗ Invalid id token(s): {', '.join(invalid)}")
        if not ids:
            print("  ✗ Usage: approve all OR approve 1,3")
            return False
        for id_ in ids:
            if id_ in pool:
                approved[id_] = pool[id_]
                print(f"  ✓ Approved [{id_}] {pool[id_]['subtopic']}")
            else:
                print(f"  ✗ No subtopic with id {id_}")
        return False

    if lowered.startswith("reject"):
        rest = cmd[len("reject"):].strip()
        ids, invalid = _parse_ids(rest)
        if invalid:
            print(f"  ✗ Invalid id token(s): {', '.join(invalid)}")
        if not ids:
            print("  ✗ Usage: reject 2 OR reject 1,3")
            return False
        for id_ in ids:
            removed_a = approved.pop(id_, None)
            removed_p = pool.pop(id_, None)
            target = removed_a or removed_p
            if target:
                print(f"  ✗ Rejected [{id_}] {target['subtopic']}")
            else:
                print(f"  ✗ No subtopic with id {id_}")
        return False

    if lowered.startswith("modify"):
        rest = cmd[len("modify"):].strip()
        m = re.match(r"(\d+)\s+(.*)", rest)
        if not m:
            print('  ✗ Usage: modify <id> "new name"')
            return False
        id_ = int(m.group(1))
        new_name = _extract_quoted(m.group(2)) or m.group(2).strip()
        if not new_name:
            print('  ✗ Usage: modify <id> "new name"')
            return False
        if id_ not in pool:
            print(f"  ✗ No subtopic with id {id_}")
            return False
        pool[id_]["subtopic"] = new_name
        if id_ in approved:
            approved[id_]["subtopic"] = new_name
        print(f"  ✎ Renamed [{id_}] to '{new_name}'")
        return False

    if lowered.startswith("add"):
        rest = cmd[len("add"):].strip()
        name = _extract_quoted(rest) or rest
        if not name:
            print('  ✗ Usage: add "new subtopic"')
            return False
        if len(name) > 120:
            print("  ✗ Subtopic too long (max 120 chars).")
            return False
        normalized_names = {item["subtopic"].strip().lower() for item in pool.values()}
        if name.strip().lower() in normalized_names:
            print(f"  ⚠︎ Subtopic '{name}' already exists. Skipping add.")
            return False
        new_id = next_id[0]
        next_id[0] += 1

        # Run a real Tavily search so the user-added subtopic has the same
        # shape as investigator-produced ones: real summary, real sources.
        # This is the fix for the "Curator returned unparseable JSON" bug
        # — the curator used to choke on empty-summary / empty-sources items.
        print(f"  🌐 Searching to enrich '{name}'...")
        results = quick_search(name, max_results=4)
        synth   = synthesize_subtopic_from_results(name, results)

        item = {
            "id":         new_id,
            "subtopic":   name,
            "summary":    synth["summary"],
            "key_points": synth["key_points"],
            "sources":    synth["sources"],
        }
        pool[new_id] = item
        approved[new_id] = item
        print(
            f"  + Added [{new_id}] {name} (auto-approved, "
            f"{len(synth['sources'])} sources)"
        )
        return False

    print(f"  ? Unknown command: {cmd}")
    return False


def _parse_user_input(findings: list[dict], raw: str) -> list[dict]:
    """Apply a pipe-chained command string to the findings pool."""
    pool: dict[int, dict] = {item["id"]: dict(item) for item in findings}
    approved: dict[int, dict] = {}
    next_id = [max(pool.keys(), default=0) + 1]

    commands = [part.strip() for part in (raw or "").split("|") if part.strip()]
    for part in commands[:20]:
        if _apply_command(part, pool, approved, next_id):
            break

    # Safety net: if the user submitted nothing meaningful, approve everything
    # rather than sending an empty list to the curator.
    if not approved:
        print("  ⚠️  No explicit approvals parsed — approving all as fallback.")
        approved = dict(pool)

    return [approved[id_] for id_ in sorted(approved.keys())]


# ─────────────────────────────────────────────────────────────────────────
# The node itself
# ─────────────────────────────────────────────────────────────────────────

def human_validation_node(state: ResearchState) -> Command:
    """
    LangGraph Human-in-the-Loop
    ───────────────────────────
    `interrupt()` PAUSES graph execution and surfaces a value to the caller
    (the `value` arg below). The graph is RESUMED by re-invoking
    graph.stream() / graph.invoke() with a `Command(resume=<user_value>)`
    object. The checkpointer (MemorySaver in this project) persists the
    full graph state between the pause and the resume.

    Fundamentally different from a normal node: instead of returning a
    plain dict, this returns a `Command`, which lets us BOTH update state
    AND tell the graph where to go next (`goto`). That removes the need
    for an explicit edge out of this node in the graph builder.

    Why this matters: in a web app, the `interrupt()` value is what you'd
    send to the frontend ("here are the findings, user — review them"),
    and the `Command(resume=...)` is what you'd send back once the user
    clicks a button. The graph itself doesn't care whether the human is
    on the CLI or across the internet.
    """
    findings = state["raw_findings"]

    # Render first, so the user can see the findings BEFORE we pause.
    _render_findings(state["topic"], findings)
    _print_commands()

    # PAUSE HERE. When the graph is re-invoked with Command(resume=<str>),
    # that string is returned from this call.
    user_input = interrupt("Waiting for human validation")

    # `user_input` is whatever the caller passed as Command(resume=...).
    # In this project it's the raw command line from the operator.
    approved = _parse_user_input(findings, user_input)
    print(f"\n  → {len(approved)} subtopic(s) approved. Proceeding.\n")

    # Command(update=..., goto=...) bundles a state update with a routing
    # decision. It is why there is no explicit edge out of this node
    # in graph_builder.py — the routing lives here.
    return Command(
        update={"approved_subtopics": approved},
        goto="curator",
    )

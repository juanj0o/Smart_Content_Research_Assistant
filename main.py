"""
Entry point. The supervisor is tiny here — LangGraph is the engine, this
file just drives it.

─────────────────────────────────────────────────────────────────────────────
Two-phase invoke pattern for human-in-the-loop
─────────────────────────────────────────────────────────────────────────────
Because `human_validation_node` calls `interrupt()`, the graph pauses
midway through. We run it in TWO phases:

  Phase 1: graph.stream(initial_state, thread_config)
           → runs investigator, enters human_validation, hits interrupt(),
             returns control here.

  (the console prompts the user)

  Phase 2: graph.stream(Command(resume=user_input), thread_config)
           → LangGraph looks up the paused run via thread_id, returns
             `user_input` from the `interrupt()` call, then continues
             through curator and reporter until END.

This is the shape of every human-in-the-loop LangGraph app — even the
ones with web UIs. The difference is just where `user_input` comes from.
"""

import os
import re
from datetime import datetime

from dotenv import load_dotenv
from langgraph.types import Command

from config import MODEL_DISPLAY_NAMES
from graph_builder import build_graph


SEP_HEAVY = "══════════════════════════════════"
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def main() -> None:
    load_dotenv()
    graph = build_graph()

    print()
    print(SEP_HEAVY)
    print("  SMART CONTENT RESEARCH ASSISTANT (LangGraph)")
    print(SEP_HEAVY)
    print()

    try:
        topic = input("Enter a research topic: ").strip()
    except EOFError:
        topic = ""
    if not topic:
        print("No topic provided. Exiting.")
        return

    # thread_config identifies a "conversation thread" for the checkpointer.
    # All state snapshots for this run are keyed by thread_id. Two parallel
    # research runs with different thread_ids would each have their own
    # independent, resumable state.
    thread_config = {"configurable": {"thread_id": "research-1"}}

    # Initial state mirrors the TypedDict schema. Fields we don't set up
    # front still need to exist as empty so reducers have something to
    # append to.
    initial_state = {
        "topic": topic,
        "raw_findings": [],
        "search_results": [],      # populated by investigator after Tavily searches
        "approved_subtopics": [],
        "curated_content": {},
        "final_report": "",
        "cost_log": [],
    }

    print(f"\n[1/4] 🔍 Investigator Agent running...")
    print(f"[2/4] 👤 Human validation required")

    # ── Phase 1: run until interrupt ───────────────────────────────────
    # stream_mode="values" yields the full accumulated state after every
    # node. We iterate to consume the stream; the graph pauses itself
    # when it hits interrupt() inside human_validation_node.
    try:
        for _event in graph.stream(initial_state, thread_config, stream_mode="values"):
            pass
    except Exception as e:
        print(f"  ⚠️  Graph error before human step: {e}")
        return

    # At this point the graph is paused. Grab the raw command line from
    # the operator. Parsing happens INSIDE the node on resume.
    user_commands = _collect_user_input()

    print(f"\n[3/4] 🧠 Curator Agent analyzing...")
    print(f"[4/4] 📝 Reporter Agent writing...")

    # ── Phase 2: resume with user input ────────────────────────────────
    # Command(resume=...) tells LangGraph: "find the paused run for this
    # thread_id, and return this value from the interrupt() call." The
    # graph then continues from that exact point.
    final_state = None
    try:
        for event in graph.stream(
            Command(resume=user_commands),
            thread_config,
            stream_mode="values",
        ):
            final_state = event
    except Exception as e:
        print(f"  ⚠️  Graph error after resume: {e}")
        return

    if final_state and final_state.get("final_report"):
        _save_report(topic, final_state["final_report"], final_state)
        _print_cost_summary(final_state.get("cost_log", []))
    else:
        print("  ⚠️  Graph finished without a final_report.")
        if final_state:
            _print_cost_summary(final_state.get("cost_log", []))


def _collect_user_input() -> str:
    """Collect the raw command string. Parsing lives inside the human node."""
    try:
        return input("> ").strip()
    except EOFError:
        return ""


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "report"


def _display(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def _save_report(topic: str, report: str, final_state: dict) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    n_sections = len(final_state.get("approved_subtopics") or [])
    models_used = sorted({
        _display(entry["model"]) for entry in final_state.get("cost_log", [])
    })
    footer = (
        f"\n\n---\n"
        f"*Report generated by Research Assistant (LangGraph) | "
        f"Sections analyzed: {n_sections} | "
        f"Models used: {', '.join(models_used)}*\n"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"{_slugify(topic)}_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + footer)
    print(f"✅ Report saved to {os.path.relpath(path)}")


def _print_cost_summary(cost_log: list[dict]) -> None:
    if not cost_log:
        return
    print()
    print(SEP_HEAVY)
    print("  COST SUMMARY")
    print(SEP_HEAVY)
    header = f"{'Agent':<14}{'Model':<14}{'Tokens In':>12}{'Tokens Out':>14}{'Cost':>12}"
    print(header)
    print("─" * len(header))

    total_in = total_out = 0
    total_cost = 0.0
    for entry in cost_log:
        agent = entry.get("agent", "?")
        model = _display(entry.get("model", "")).replace("claude-", "")
        in_t = int(entry.get("input_tokens", 0))
        out_t = int(entry.get("output_tokens", 0))
        cost = float(entry.get("cost_usd", 0.0))
        total_in += in_t
        total_out += out_t
        total_cost += cost
        print(f"{agent:<14}{model:<14}{in_t:>12,}{out_t:>14,}{'$' + f'{cost:.4f}':>12}")

    print("─" * len(header))
    print(f"{'TOTAL':<28}{total_in:>12,}{total_out:>14,}{'$' + f'{total_cost:.4f}':>12}")
    print(SEP_HEAVY)


if __name__ == "__main__":
    main()

"""
Where the graph is assembled.

─────────────────────────────────────────────────────────────────────────────
StateGraph concepts at a glance
─────────────────────────────────────────────────────────────────────────────
- Nodes             Units of work. Plain functions (or callables) with the
                    signature `(state) -> dict | Command`.
- Edges             Static routing between nodes: "after A, run B".
- Conditional edges Dynamic routing based on a predicate over state.
- START / END       Special LangGraph sentinels marking entry and exit.
- compile()         Validates the graph and returns a runnable object.
- checkpointer      Persists state snapshots — REQUIRED for `interrupt()`
                    so the graph can be paused and later resumed.

Contrast with the from-scratch version: there, a `Supervisor` class called
each agent in a fixed sequence — orchestration logic was hand-written. Here,
the graph engine owns execution order. Nodes don't know about each other;
they only know about the shared state.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph_state import ResearchState
from nodes.curator_node import curator_node
from nodes.human_node import human_validation_node
from nodes.investigator_node import investigator_node
from nodes.reporter_node import reporter_node


def build_graph():
    # StateGraph is parameterized with our TypedDict. This gives LangGraph
    # the schema it needs to validate node return values, apply reducers,
    # and serialize state to the checkpointer.
    builder = StateGraph(ResearchState)

    # ── Nodes ────────────────────────────────────────────────────────────
    # Each call registers a named unit of work. The name is what edges
    # and Command(goto=...) reference.
    builder.add_node("investigator", investigator_node)
    builder.add_node("human_validation", human_validation_node)
    builder.add_node("curator", curator_node)
    builder.add_node("reporter", reporter_node)

    # ── Edges ────────────────────────────────────────────────────────────
    # START is a sentinel meaning "entry point". The first real node is
    # whatever we edge START to.
    builder.add_edge(START, "investigator")
    builder.add_edge("investigator", "human_validation")
    # NOTE: no edge out of "human_validation" — that node returns a
    # Command(goto="curator"), which routes dynamically from inside the
    # node itself. This is why Command is powerful: it lets individual
    # nodes make routing decisions without the graph builder knowing.
    builder.add_edge("curator", "reporter")
    builder.add_edge("reporter", END)

    # ── Checkpointer ─────────────────────────────────────────────────────
    # MemorySaver keeps state snapshots in RAM. It is REQUIRED for
    # interrupt() to work: the graph has to serialize its state when it
    # pauses, then rehydrate when resumed. For production you'd swap in
    # PostgresSaver or RedisSaver — same interface, durable storage.
    checkpointer = MemorySaver()

    # compile() validates the graph (every referenced node exists, no
    # unreachable nodes, etc.) and returns a CompiledGraph we can invoke.
    return builder.compile(checkpointer=checkpointer)

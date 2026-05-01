# Research Assistant — LangGraph Edition

A console-based multi-agent research assistant built with **LangGraph** and **real web search** (Tavily). Designed to be pedagogically explicit: every LangGraph concept is visible in the code with inline comments.

---

## Architecture

```
User input (topic)
      │
      ▼
┌─────────────────┐     3 concurrent     ┌──────────────┐
│  Investigator   │──── Tavily searches ─▶│  Web Results │
│  (fast model)   │◀────────────────────  └──────────────┘
└────────┬────────┘
         │ raw_findings (4–6 subtopics + real URLs)
         ▼
┌─────────────────┐
│ Human Validation│  approve / reject / modify / add
│  (interrupt)    │
└────────┬────────┘
         │ approved_subtopics
         ▼
┌─────────────────┐
│    Curator      │  deep analysis, relevance scores, cross-links
│  (smart model)  │
└────────┬────────┘
         │ curated_content (with sources preserved)
         ▼
┌─────────────────┐
│    Reporter     │  polished markdown report + References section
│ (premium model) │
└────────┬────────┘
         │
         ▼
  reports/{topic}_{timestamp}.md
```

### Key LangGraph concepts used

| Concept | Where | Why |
|---|---|---|
| `TypedDict` state | `graph_state.py` | Shared, serializable state across all nodes |
| `Annotated[list, operator.add]` | `graph_state.py` | Reducer — appends instead of replacing |
| `StateGraph` | `graph_builder.py` | Declarative graph assembly |
| `interrupt()` | `nodes/human_node.py` | Pauses execution for human input |
| `Command(update, goto)` | `nodes/human_node.py` | State update + dynamic routing in one |
| `MemorySaver` | `graph_builder.py` | Checkpointer — persists state across interrupt |
| `LCEL` pipe `\|` | Every node | `prompt \| llm` = RunnableSequence |

---

## Project Structure

```
research_assistant_langgraph/
├── main.py                  # Entry point — drives the two-phase graph execution
├── config.py                # Model tiers (fast / smart / premium) per provider
├── graph_state.py           # ResearchState TypedDict — the core of LangGraph
├── graph_builder.py         # StateGraph assembly: nodes, edges, checkpointer
├── llm_factory.py           # Returns ChatOllama or ChatGroq based on .env
├── cost_tracker.py          # Token usage helpers
├── json_utils.py            # Robust JSON parser (handles truncation, fences, etc.)
├── search_helper.py         # Shared Tavily helpers (used by human_node add command)
├── nodes/
│   ├── investigator_node.py # Web search + LLM synthesis (concurrent via ThreadPool)
│   ├── curator_node.py      # Deep analysis + source preservation
│   ├── reporter_node.py     # Markdown report + deterministic References section
│   └── human_node.py        # interrupt() + Command — human-in-the-loop
├── tests/
│   ├── test_json_utils.py   # Parser edge cases
│   └── test_human_node.py   # Command parsing (approve / reject / modify / add)
├── reports/                 # Generated reports saved here (auto-created)
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd research_assistant_langgraph
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Choose your LLM provider
LLM_PROVIDER=groq          # or: ollama

# Groq API key (required if LLM_PROVIDER=groq) — free at console.groq.com
GROQ_API_KEY=gsk_...

# Tavily API key (required for web search) — free tier: 1,000 searches/month
# Get it at: https://app.tavily.com/
TAVILY_API_KEY=tvly-...
```

> **Using Ollama?** Make sure Ollama is running locally (`ollama serve`) and the models are pulled:
> ```bash
> ollama pull qwen2.5:7b && ollama pull qwen2.5:32b
> ```

### 3. Run

```bash
python main.py
```

---

## Usage

```
Enter a research topic: quantum computing

[1/4] 🔍 Investigator Agent running...
    🌐 Iniciando búsquedas concurrentes para: quantum computing
    ...
[2/4] 👤 Human validation required

══════════════════════════════════════════════
  RESEARCH FINDINGS — Human Validation Required
══════════════════════════════════════════════

Topic: quantum computing

Subtopics found:
  [1] Quantum Hardware
  [2] Error Correction
  [3] Quantum Algorithms
  [4] Real-world Applications
  [5] Quantum vs Classical

Commands:
  approve all              → approve everything
  approve 1,3              → approve specific items
  reject 2                 → remove an item
  modify 1 "new name"      → rename a subtopic
  add "new subtopic"       → add a custom subtopic (triggers a live search)
  done                     → finish editing

You can chain commands: "approve 1,3 | reject 2 | add 'Quantum cryptography'"
> approve all

[3/4] 🧠 Curator Agent analyzing...
[4/4] 📝 Reporter Agent writing...
✅ Report saved to reports/quantum_computing_20260501_184629.md
```

The generated report includes:
- Executive Summary, Introduction
- One section per approved subtopic with inline citations `[1][2]`
- Cross-cutting Insights and Conclusions
- A `## References` section grouped by subtopic, built deterministically in Python

---

## How the two-phase interrupt works

```python
# Phase 1 — runs until interrupt() inside human_validation_node
for event in graph.stream(initial_state, thread_config, stream_mode="values"):
    pass   # graph pauses here

# user types commands in the console

# Phase 2 — resume, passing user input back into the graph
for event in graph.stream(Command(resume=user_commands), thread_config, ...):
    final_state = event
```

The `MemorySaver` checkpointer serializes the full graph state between phases. In a production app you'd replace it with `PostgresSaver` and drive Phase 2 from an HTTP endpoint.

---

## Running tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Providers & Models

| Provider | Fast (Investigator) | Smart (Curator) | Premium (Reporter) |
|---|---|---|---|
| `groq` | llama-3.1-8b-instant | llama-3.3-70b-versatile | llama-3.3-70b-versatile |
| `ollama` | qwen2.5:7b | qwen2.5:32b | qwen2.5:32b |

Switch providers by changing `LLM_PROVIDER` in `.env`. No code changes needed.

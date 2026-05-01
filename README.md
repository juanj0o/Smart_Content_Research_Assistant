# Research Assistant — LangGraph Edition

A console-based multi-agent research assistant built with **LangGraph**, **real web search** (Tavily), and a source credibility scorer. 

---

## Architecture

```
User input (topic)
      │
      ▼
┌─────────────────────────────────────────────┐
│  Investigator  (fast model)                 │
│  ├─ 3 concurrent Tavily searches            │
│  ├─ Source credibility scoring + ranking    │
│  └─ LLM synthesis → 4-6 subtopics + URLs   │
└────────────────────┬────────────────────────┘
                     │ raw_findings
                     ▼
┌─────────────────────────────────────────────┐
│  Human Validation  (interrupt)              │
│  approve / reject / modify / add            │
│  └─ add triggers a live Tavily search       │
└────────────────────┬────────────────────────┘
                     │ approved_subtopics
                     ▼
┌─────────────────────────────────────────────┐
│  Curator  (smart model)                     │
│  Deep analysis, relevance scores,           │
│  cross-links, sources preserved             │
└────────────────────┬────────────────────────┘
                     │ curated_content
                     ▼
┌─────────────────────────────────────────────┐
│  Reporter  (premium model)                  │
│  Markdown report with inline citations [N]  │
│  References section built in Python         │
└────────────────────┬────────────────────────┘
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
| LCEL pipe `\|` | Every node | `prompt \| llm` = RunnableSequence |

---

## Project Structure

```
research_assistant_langgraph/
├── main.py                   # Entry point — drives the two-phase graph execution
├── config.py                 # Model tiers (fast / smart / premium) per provider
├── graph_state.py            # ResearchState TypedDict — the core of LangGraph
├── graph_builder.py          # StateGraph assembly: nodes, edges, checkpointer
├── llm_factory.py            # Returns ChatOllama or ChatGroq based on .env
├── cost_tracker.py           # Token usage helpers
├── json_utils.py             # Robust JSON parser (handles truncation, fences, etc.)
├── search_helper.py          # Tavily helpers shared by investigator and human node
├── credibility_scorer.py     # Domain-based source quality scoring and ranking
├── nodes/
│   ├── investigator_node.py  # Concurrent web search + credibility filter + LLM
│   ├── curator_node.py       # Deep analysis + source preservation
│   ├── reporter_node.py      # Markdown report + deterministic References section
│   └── human_node.py         # interrupt() + Command — human-in-the-loop
├── reports/                  # Generated reports saved here (auto-created)
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
    📊 Credibilidad: 12 sources — 🟢 5 high  🟡 6 medium  🔴 1 low
[2/4] 👤 Human validation required

══════════════════════════════════════════════
  RESEARCH FINDINGS — Human Validation Required
══════════════════════════════════════════════

Topic: quantum computing

Subtopics found:
  [1] Quantum Hardware
      Summary: ...
      Key points: qubit stability, superconducting circuits, error rates

  [2] Quantum Algorithms
  [3] Error Correction
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
- Executive Summary and Introduction
- One section per approved subtopic with inline citations `[1][2]`
- Cross-cutting Insights and Conclusions
- A `## References` section grouped by subtopic, built deterministically in Python (never by the LLM)

---

## Source Credibility Scoring

The `credibility_scorer.py` module scores every Tavily result from `0.0` to `1.0` before passing them to the LLM:

- **Positive signals**: `.gov` / `.edu` TLDs, known scientific journals (PMC, Nature, arXiv, IEEE), health organizations (WHO, CDC), trusted news outlets (Reuters, BBC)
- **Negative signals**: commercial URL patterns (`/shop`, `/buy`), product-selling content phrases, clickbait titles
- Results are **sorted by score descending** — high-quality sources appear first in the LLM prompt (primacy bias)
- Results below `0.15` are filtered out; a minimum of 5 results is always kept

The console shows a summary per run: `📊 Credibilidad: 12 sources — 🟢 5 high 🟡 6 medium 🔴 1 low`

> **Known limitation**: the scorer reduces the influence of low-quality sources but does not eliminate them entirely when they are the only sources available for a topic. Topics with high misinformation density (e.g. health conspiracy theories) may still surface biased content — the output should be reviewed critically. A plaussible solution is to incorporate an agent to the workflow in charge of scoring the sources.
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

## Providers & Models

| Provider | Fast (Investigator) | Smart (Curator) | Premium (Reporter) |
|---|---|---|---|
| `groq` | llama-3.1-8b-instant | llama-3.3-70b-versatile | llama-3.3-70b-versatile |
| `ollama` | qwen2.5:7b | qwen2.5:32b | qwen2.5:32b |

Switch providers by changing `LLM_PROVIDER` in `.env`. No code changes needed.

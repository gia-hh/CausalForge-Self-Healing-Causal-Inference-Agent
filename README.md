# Automated Causal Inference Expert System

A LangGraph state machine that plans a causal-inference design, generates and
self-repairs statistical analysis code, **enforces statistical diagnostics on the
host side** (not via the LLM), and produces a validated business report. The whole
system is verified end-to-end against a synthetic dataset with a **known
ground-truth ATE (τ = 0.05)**.

This is an MVP designed for a single laptop: in-memory state, no Docker, optional
fully-offline operation.

---

## Why this design

| Goal | How it is implemented |
|------|-----------------------|
| Parse the causal structure | Planning node + LangGraph `interrupt` HITL |
| Generate executable statistics code | CodeGen node (heavy LLM), emits stratified ATE |
| Self-repair + loop protection | Repair sub-loop via conditional edges + a circuit breaker |
| Statistical assumptions checked by the host | Executor node runs native VIF / Breusch-Pagan |
| Cost control | Asymmetric LLM routing (heavy vs light) |
| Verifiable results | Binary-CTR synthetic data + god's-eye ground truth |

Key principles:

* Every agent is a **LangGraph node** that mutates a shared `AgentState`.
* The repair loop is built from **conditional edges**, not a `while` loop, so the
  graph is fully visualizable.
* The **LLM only chooses methods and fits a model**; the host Python enforces the
  diagnostics (VIF, Breusch-Pagan) so they can never be "forgotten".
* Heterogeneity (HTE) strata are estimated with the **same IPW weights** as the
  main effect — never degraded to a raw T-test.

---

## Project layout

```
causal_agent/
├── run_pipeline.py            # single entry point (runner)
├── requirements.txt           # pinned deps for Python 3.10.11
├── README.md
├── tests/
│   └── test_acceptance.py     # MVP acceptance tests (offline, deterministic)
└── src/causal_agent/
    ├── config.py              # models, backend switch, thresholds
    ├── state.py               # AgentState TypedDict + reducers
    ├── data.py                # synthetic data + metadata
    ├── executor.py            # sandbox exec + host-side VIF/BP diagnostics
    ├── runtime.py             # holds the DataFrame outside graph state
    ├── graph.py               # LangGraph assembly (nodes + edges)
    ├── llm/
    │   ├── prompts.py         # all system prompts (English)
    │   ├── mock_client.py     # deterministic offline responses
    │   └── client.py          # mock / Anthropic / Ollama dispatcher
    ├── nodes/
    │   ├── planning.py        # Node 1 (heavy + interrupt HITL)
    │   ├── codegen.py         # Node 2 (heavy; optional fault injection)
    │   ├── executor_node.py   # Node 3 (host)
    │   ├── parsers.py         # Nodes 4a/4b + routers
    │   ├── repair.py          # Node 5 (heavy; blacklist-aware)
    │   ├── hte.py             # Node 6 (light; pure interpretation)
    │   ├── sanity.py          # Node 7 (light + host re-verification)
    │   └── report.py          # Node 8 + human escalation
    └── utils/
        ├── json_utils.py      # robust JSON / code-fence extraction
        ├── validators.py      # all pure-Python self-evaluations
        └── repair_utils.py    # blacklist, repair context, rule parser
```

---

## The workflow

```
query + metadata
      │
      ▼
  Planning ──unclear vars?──► interrupt()  ◄─ Command(resume=decisions)
      │ experiment_design (with outcome_type)
      ▼
  CodeGen  (emits results_dict + stratified_results + model_fit)
      │
      ▼
  Executor  ──► 1. exec(code) in isolated namespace
      │         2. extract model_fit → host VIF / BP diagnostics
      │         3. extract results_dict → numeric validation
      │
      ├─ success ──► HTE (interpretation) ──► Sanity ──► Report ──► END
      │
      └─ failure ──► Error Router
                       ├─ syntactic   → Rule-based Parser ─┐
                       └─ semantic     → LLM Parser ────────┤
                                                            ▼
                            repair_attempts ≥ 3 ? ──► Human Escalation
                                                            │ no
                                                            ▼
                                                  Repair ──► (back to Executor)
```

---

## Setup (Python 3.10.11)

```bash
# 1. Create and activate a 3.10.11 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` are mutually compatible and ship cp310
wheels, so the install needs no compiler.

---

## Running

### Offline (default, no API key needed)

The default backend is `mock`: deterministic, fully offline, reproducible.

```bash
python run_pipeline.py
```

Expected tail of the output:

```
[ACCEPTANCE CHECK vs ground truth tau=0.05]
  estimated ATE = 0.0416  (target [0.04, 0.06]) -> PASS
  p-value       = 0.0006  (target < 0.05)        -> PASS
  repair attempts used = 0
```

### Fault-injection demos (exercise the self-repair loop)

```bash
# Injects a NameError -> routed to the rule-based parser -> repaired
INJECT_FAULT=nameerror python run_pipeline.py

# Injects perfect multicollinearity -> host VIF check raises
# StatisticalAssumptionError -> routed to the LLM parser -> repaired
INJECT_FAULT=multicollinearity python run_pipeline.py
```

Both recover within the 3-attempt budget and still produce a valid report.

### Real backends

Heavy nodes → Anthropic API; light nodes → local Ollama.

```bash
# Start Ollama and pull the light model first:
#   ollama serve
#   ollama pull llama3:8b

export LLM_BACKEND=real
export ANTHROPIC_API_KEY=sk-ant-...        # required for heavy nodes
export OLLAMA_HOST=http://localhost:11434  # optional, this is the default
python run_pipeline.py
```

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_BACKEND` | `mock` | `mock` (offline) or `real` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_BACKEND=real` (heavy nodes) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL (light nodes) |
| `HEAVY_MODEL` | `claude-sonnet-4-6` | Heavy reasoning model ID |
| `LIGHT_MODEL_OLLAMA` | `llama3:8b` | Light reasoning Ollama model |
| `INJECT_FAULT` | — | `nameerror` or `multicollinearity` (demos) |

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
# or, with no extra dependency:
python tests/test_acceptance.py
```

The suite mirrors the MVP acceptance table: ATE recovery, confounding correction,
HTE direction, both repair paths, the VIF diagnostic, and the BP/robust-SE
interaction.
```

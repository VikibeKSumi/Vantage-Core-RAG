# PolicyLens — Master Refactor Plan

> **Working name:** `PolicyLens` *(descriptive, technical — "lens" = focused examination of policy. Easy to rename.)*

---

## 1. Context — Why We're Doing This

The current project (`vantage_core_rag`) is a well-architected **MVP** for querying 2 Indian economic PDFs. It has solid bones (shared embedding model, two-stage retrieval, LlamaIndex + Qdrant) but is fundamentally limited:

- **Tiny corpus** — locked to Budget + Economic Survey
- **15+ hardcoded values** scattered across files (collection name, top-k, thresholds, prompts)
- **No persistent cache** — in-memory, lost on restart
- **No real auth, fake metrics, single LLM provider** — not production
- **No evaluation infrastructure** — 14 hardcoded queries ≠ reliability

**What we're building:** A natural-language research assistant for India's policy/regulatory/government corpus — letting users ask in English/Hindi and get accurate, cited answers with multi-step reasoning.

**Why this domain:** Real audience (lawyers, journalists, businesses, citizens), freely available data, natural extension of existing work, massive unmet need.

---

## 2. Project Identity

| | |
|---|---|
| **Name** | PolicyLens |
| **Tagline** | *"Clarity through India's policy maze"* |
| **Aim** | Turn scattered, dense government documents into instant, cited answers |
| **Repo target** | `policy-lens/` (rename from `vantage_core_rag`) |

### Target Users
- 🏛️ Lawyers / legal researchers
- 📰 Journalists / policy analysts
- 🏢 Businesses (compliance teams)
- 🌱 NGOs / civil society
- 🎓 UPSC / civil service aspirants
- 👥 Engaged citizens

### What It Answers
| Level | Example |
|---|---|
| Factual | *"What's the FY27 fiscal deficit target?"* |
| Comparative | *"Compare defense vs education spending over 5 years"* |
| Cross-document | *"Are RBI fintech guidelines aligned with SEBI's?"* |
| Analytical (Agentic) | *"Summarize all EV-related policy changes since 2023"* |
| Relationship (GraphRAG) | *"Trace policy: cabinet decision → ministry notification → gazette"* |

### What It WON'T Do
- Personal financial/legal advice
- Future predictions
- Real-time data
- Anything outside the indexed corpus

---

## 3. Coaching Model — How We Work

| AI does | User does |
|---|---|
| Writes specs ("build a function that...") | Writes ALL project code |
| Explains *why* before *how* | Asks why when stuck |
| Reviews user's code, points issues | Tests, breaks, fixes |
| Breaks big steps into smaller ones | Drives architectural choices |
| Teaches concepts before implementation | Learns by building |
| Edits ONLY the plan file | Owns the codebase |

**Workflow per step:**
1. AI gives spec + explains why
2. User attempts implementation
3. User shows AI the code
4. AI reviews → approves or suggests changes
5. Iterate until clean
6. Move to next step

---

## 4. Target Architecture

```
                      ┌─────────────────────┐
                      │  User Query (UI/API)│
                      └──────────┬──────────┘
                                 ↓
                    ┌────────────────────────┐
                    │   Auth + Rate Limiting │
                    └──────────┬─────────────┘
                               ↓
                    ┌──────────────────────┐
                    │  Persistent Cache    │  ← Redis
                    └──────────┬───────────┘
                               ↓ (miss)
                    ┌──────────────────────┐
                    │   AGENTIC ROUTER     │
                    │  - Query classifier  │
                    │  - Strategy selector │
                    └──────────┬───────────┘
                               ↓
            ┌──────────────────┼──────────────────┐
            ↓                  ↓                  ↓
   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
   │ Vector RAG     │ │ Graph RAG      │ │ Tools          │
   │ (Qdrant)       │ │ (Neo4j/NetX)   │ │ (calc, fetch)  │
   └────────┬───────┘ └────────┬───────┘ └────────┬───────┘
            └──────────────────┼──────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │  Reranker + Compress │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  Generator           │
                    │  Groq → vLLM fallback│
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  Response + Citations│
                    │  + Logged for eval   │
                    └──────────────────────┘
```

---

## 5. Tech Stack

### Core RAG
| Layer | Tool | Status |
|---|---|---|
| Orchestration | LlamaIndex | ✅ keep |
| Vector DB | Qdrant (server mode) | ✅ keep, upgrade |
| Graph DB | Neo4j *(or NetworkX in dev)* | ➕ add Phase 4 |
| Embeddings | BAAI/bge-small + IndicBERT | ✅ keep, upgrade later |
| Reranker | BAAI/bge-reranker-v2-m3 | ✅ keep, upgrade |
| LLM (primary) | Groq (Llama 3.3 70B) | ✅ keep |
| LLM (fallback) | vLLM self-hosted / GPT-4o-mini | ➕ add Phase 2 |

### Agentic & Data
| Tool | Purpose |
|---|---|
| LangGraph **or** LlamaIndex Agents | Agent state machine |
| `unstructured.io` | Parse PDFs, tables, images |
| Scrapy / Playwright | Crawl government portals |
| `datasketch` (MinHash LSH) | Deduplication |
| `tiktoken` | Real token counting |

### Production Infra
| Tool | Purpose |
|---|---|
| Redis | Persistent cache + rate limiting |
| FastAPI | Real API |
| `pydantic-settings` v2 | Typed config |
| OpenTelemetry | Distributed tracing |
| Prometheus + Grafana | Metrics + dashboards |
| Docker Compose | 6-service orchestration |
| GitHub Actions | CI/CD with quality gates |

### Evaluation (CRITICAL)
| Tool | Purpose |
|---|---|
| **RAGAS** | Faithfulness, relevancy, context precision/recall |
| **DeepEval** | Complementary framework |
| **MLflow** | Track eval runs, compare experiments |
| **pytest** | Run eval as test suite |
| HuggingFace `datasets` | Version golden Q&A set |

### Auth
| Tool | Purpose |
|---|---|
| FastAPI-Users / `python-jose` | JWT auth |
| `slowapi` | Rate limiting |
| Pydantic | Input validation |

---

## 6. Critical Bugs to Fix First (Phase 0)

These exist in the current codebase and block any clean refactor:

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `src/cache.py` | Loads own SentenceTransformer (model duplication) | Inject shared embedder |
| 2 | `src/cache.py` | No eviction → unbounded memory | Replace with Redis + TTL |
| 3 | `src/compressor.py:24` | `text.split()` ≠ tokens (2-3x error) | Use `tiktoken` |
| 4 | `src/config.py` | No schema validation | Pydantic v2 Settings |
| 5 | `src/retrieve_and_rerank.py:12` | `top_k=4` hardcoded | Read from settings |
| 6 | `src/engine_load.py:157` | `max_tokens=1500` hardcoded | Read from settings |
| 7 | `src/engine_load.py:68` | Cache threshold `0.85` hardcoded | Read from settings |
| 8 | `src/generation.py:35-40` | Prompt hardcoded in Python | Move to Jinja2 template |
| 9 | `src/logger.py:6` | Log path hardcoded | Read from settings |
| 10 | `api.py` | Metrics fake/hardcoded | Real Prometheus counters |
| 11 | `Dockerfile:38` | Healthcheck hits non-existent `/healthz` | Fix to `/health` |
| 12 | `ingestion.py` | No error recovery | Per-stage try/except |
| 13 | `src/database.py` | Local Qdrant only | Env-aware client |
| 14 | `.github/workflows/ci_pipeline.yaml` | No tests before build | Add lint+test+eval gate |
| 15 | `run.py` | Typo "type 'exit' or 'quit' end" | "to end" |

---

## 7. Phased Roadmap

> Each phase: **Goal → Your Steps → What You Learn → Eval Checkpoint → Done When**

---

### 🏗️ Phase 1 — Foundation Reset *(Week 1-2)*

**Goal:** New project skeleton + minimum eval baseline before changing logic.

**Your steps:**
1. Create new repo structure (`policy-lens/`)
2. Migrate config to Pydantic Settings v2 (typed, validated, env-overridable)
3. Set up env-driven logging
4. Fix all 15 critical bugs above
5. Write **minimum golden eval set** (50 Q&A pairs from current corpus)
6. Wire up RAGAS → score current pipeline → establish baseline

**What you'll learn:** Config validation, RAGAS metrics, ground truth design.

**Eval checkpoint:** Baseline RAGAS scores documented. *Every future change compares to these.*

**Done when:** New repo runs identically to current, baseline scored, you understand each metric.

---

### 📚 Phase 2 — Data Expansion *(Week 3-4)*

**Goal:** From 2 PDFs to 10,000+ government documents.

**Your steps:**
1. Build `unstructured.io`-based parser (handles tables, hierarchy)
2. Write Scrapy spiders: PIB, RBI, SEBI, Gazette, NITI Aayog, data.gov.in
3. Add deduplication (MinHash LSH)
4. Implement incremental ingestion (SHA-256 → skip if unchanged)
5. Add metadata schema (source, date, ministry, document type)
6. Re-ingest at scale; expand eval set to 200 queries

**What you'll learn:** Real-world data engineering pain (messy PDFs, dedup, schema).

**Eval checkpoint:** RAGAS scores on 200 queries across multiple doc types. No regression.

**Done when:** 10K+ docs indexed with clean metadata.

#### Data Sources Cheat Sheet
| Source | Scale | Method |
|---|---|---|
| `data.gov.in` | 10K+ datasets | API + Scrapy |
| `egazette.nic.in` | 50K+ PDFs | Scrapy |
| `indiabudget.gov.in` | ~500 PDFs | Direct download |
| `rbi.org.in` | 5K+ docs | Scrapy + PDF |
| `sebi.gov.in` | 10K+ docs | Scrapy + PDF |
| `pib.gov.in` | 100K+ releases | API + Scrapy |
| HF `ai4bharat/sangraha` | Multilingual | `datasets` |

---

### 🛡️ Phase 3 — Production Hardening *(Week 5-6)*

**Goal:** Actually production-grade, not just feature-complete.

**Your steps:**
1. Replace in-memory cache with Redis (persistent, evictable, TTL)
2. Add fallback LLM chain: Groq → vLLM → cached response
3. Add real auth (JWT) + rate limiting (slowapi + Redis)
4. Replace fake `/metrics` with real Prometheus metrics
5. Add OpenTelemetry tracing (end-to-end query traces)
6. Set up Grafana dashboards (P95 latency, cache hit, error rate)
7. Add CI/CD quality gate: **block merge if RAGAS drops > 2%**
8. Expand eval set to 400 queries (adversarial + multilingual)

**What you'll learn:** What separates demo from product.

**Eval checkpoint:** All scores ≥ baseline; latency P95 < 5s cached, < 25s fresh.

**Done when:** Anyone could fork + deploy without your hand-holding.

---

### 🧠 Phase 4 — Intelligence Upgrades *(Week 7-10)*

**Goal:** Single-shot RAG → agentic + graph reasoning.

**Your steps:**
1. **Agentic layer (LangGraph/LlamaIndex):** Query classifier → strategy router
2. Implement tools: calculator, sub-query decomposition, citation verifier
3. **GraphRAG:** Entity extraction → Neo4j → relationship queries
4. **Adaptive routing:** Decide per-query: vector / graph / agentic
5. Multi-modal upgrade: extract tables as structured data
6. A/B test each upgrade against baseline using full 400-query eval set

**Routing logic:**
```
Rule-based fast path:
  "compare/vs/difference"     → COMPLEX
  "calculate/how much/%"      → TOOL_USE
  short query (<20 tokens)    → SIMPLE

LLM tier (only if ambiguous): structured output → {type, sub_queries}

Execution:
  SIMPLE       → cache → retrieve → generate
  COMPLEX      → decompose → [N × SIMPLE] → synthesize
  MULTI_HOP    → ReActAgent loop (max 6 steps)
  TOOL_USE     → ToolAgent → calculator/doc-fetch
  RELATIONAL   → GraphRAG traversal
```

**What you'll learn:** Agent design patterns, graph data modeling, when each strategy wins.

**Eval checkpoint:** Multi-hop & comparative scores improve significantly; simple queries don't regress.

**Done when:** System chooses right strategy per query; previously failing complex queries succeed.

---

### 🚀 Phase 5 — Launch & Iterate *(Week 11-12)*

**Goal:** Real users, real feedback, continuous improvement.

**Your steps:**
1. Beta launch (10-20 users from target audience)
2. Collect query logs + thumbs up/down feedback
3. Weekly human eval on 50 sampled production queries
4. Add failed queries to golden eval set (it grows)
5. Iterate based on real usage
6. Public launch when quality + feedback align

**Eval checkpoint:** Continuous — weekly RAGAS, monthly human eval.

**Done when:** Users come back voluntarily, NPS > 0.

---

## 8. Evaluation Strategy *(cross-cutting, MOST IMPORTANT)*

> *Without this, every change is a guess. With this, every change is provable.*

### Five Layers
| Layer | When | What |
|---|---|---|
| **Golden Set** | Phase 0 onward | 50 → 200 → 400+ Q&A pairs by category |
| **RAGAS** | Phase 0 onward | Faithfulness, relevancy, context precision/recall |
| **LLM-as-Judge** | Phase 2 onward | Completeness, citation accuracy, tone |
| **Human Eval** | Phase 4 onward | Weekly sample of 50 real queries |
| **A/B Tests** | Phase 2 onward | Every config change runs both sides |

### Golden Set Categories (300-500 queries by Phase 4)
- Factual lookup (basic retrieval)
- Numerical extraction
- Comparative (multi-doc)
- Multi-hop (chain reasoning)
- Adversarial (out-of-corpus → must refuse)
- Indic language queries
- Temporal reasoning
- Edge cases (empty, typo, very long)

### Production Targets (Phase 2 exit)
- Faithfulness > 0.90
- Answer Relevancy > 0.85
- Context Precision > 0.80
- Context Recall > 0.85

### CI Quality Gate
```
Push → lint → typecheck → unit → integration → eval
                                                  ↓
                                        Compare to baseline
                                                  ↓
                                  Drop > 2%? BLOCK MERGE
                                                  ↓
                                          Build → Deploy
```

---

## 9. New Project Structure (target)

```
policy-lens/
├── pyproject.toml                    # uv-managed, pinned deps
├── .env.example
├── Dockerfile                        # multi-stage
├── docker-compose.yml                # 6 services
├── config/
│   ├── settings.yaml                 # base defaults
│   ├── domains/
│   │   ├── policy.yaml               # India Govt/Policy
│   │   ├── legal.yaml                # future
│   │   └── medical.yaml              # future
│   └── prompts/
│       ├── base_system.jinja2
│       └── policy_system.jinja2
│
├── src/policy_lens/
│   ├── core/                         # interfaces, schemas, exceptions
│   ├── config/                       # Pydantic Settings + domain loader
│   ├── ingestion/                    # pipeline, loaders, dedup, cleaners
│   ├── retrieval/                    # embedder, reranker, vector_store
│   ├── generation/                   # generator, llm_registry, prompts, compressor
│   ├── cache/                        # Redis-backed semantic cache
│   ├── agent/                        # router, planner, memory, tools/
│   ├── graph/                        # builder, store, retriever (Phase 3)
│   ├── engine/                       # rag_engine.py
│   ├── api/                          # FastAPI routers, auth, middleware
│   ├── observability/                # metrics, tracing, logger
│   └── evaluation/                   # ragas_eval, golden_dataset, regression
│
├── scripts/                          # CLI entry points
│   ├── ingest.py
│   ├── eval.py
│   └── run_cli.py
│
├── ui/
│   └── app.py                        # Streamlit
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── data/
│   ├── raw_docs/policy/
│   ├── qdrant_storage/               # dev only
│   └── golden/policy.jsonl           # eval set
│
└── deployments/
    ├── docker-compose.prod.yml
    └── k8s/
```

---

## 10. Verification Plan

### After Phase 0
- `python -m policy_lens.scripts.run_cli` works
- `pytest tests/unit/ -v` passes, ≥70% coverage
- `grep -r "0.85\|1500\|fiscal_budget" src/` returns nothing
- Baseline RAGAS scores logged in MLflow

### After Phase 1
- `python scripts/ingest.py --domain policy --delta` ingests 1000+ docs
- Restart app → cache survives (query → restart → query → cache hit)
- All 6 docker-compose services start healthy

### After Phase 2
- `GET /readiness` all green (Qdrant ✅ Redis ✅ Groq ✅)
- `GET /metrics` returns real Prometheus counters
- `POST /v1/query` with no auth → 403
- 200 req/min → 429 after limit
- CI pipeline blocks if eval regresses

### After Phase 3
- Simple query → fast path, <3s
- "Compare RBI policy 2023 vs 2024 and calculate rate change" → ToolAgent uses calculator
- `curl -N /v1/query/stream` → tokens stream live
- Multi-hop query returns graph-sourced context

### After Phase 4
- Real users return; weekly human eval shows improvement
- Failed production queries flow into golden set automatically

---

## 11. Open Questions for User (Decide Before Phase 0)

1. **Name lock-in:** PolicyLens, NitiCore, or something else?
2. **Domain scope:** India-only, or design multi-country from day 1?
3. **Hosting target:** Local-only? Render/Railway? AWS/GCP?
4. **Multilingual priority:** English-first then Hindi, or simultaneous?
5. **Open source vs proprietary:** Affects auth/license/branding choices.
6. **Time commitment:** ~12 weeks assumes ~10-15 hrs/week. Realistic?

---

## Summary

**Building:** PolicyLens — production-grade RAG + Agent + GraphRAG platform for Indian policy/regulatory documents.

**How:** User codes step-by-step with AI as coach. AI writes specs and reviews; AI does NOT write project code.

**Phases:** Foundation → Data Expansion → Hardening → Intelligence → Launch (~12 weeks).

**North star:** Every change must move RAGAS scores forward (or at minimum not back).

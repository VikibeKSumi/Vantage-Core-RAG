# Vantage Core-RAG — Technical Architecture Map

1. System Objective

Vantage Core-RAG is a local-first, resource-constrained RAG engine designed to deliver industrial-grade retrieval precision on consumer hardware (≈4GB VRAM / CPU-heavy).

The system prioritizes:
- deterministic retrieval quality,
- memory-aware design,
- modular extensibility,
- sovereignty of data (no external embedding or vector storage).

2. Core Architectural Principle

Two-Stage Retrieval beats single-pass RAG under constrained hardware.

Instead of relying on a single dense retrieval step (“naive RAG”), Vantage Core-RAG splits retrieval into orthogonal responsibilities:
- Stage 1 (Recall-Optimized): Fast, low-memory semantic search
- Stage 2 (Precision-Optimized): Deep neural relevance verification

This separation allows the system to:
- maximize recall cheaply,
- apply expensive reasoning only where it matters,
- remain performant on CPU-only environments.

3. High-Level System Flow
               ┌────────────────────┐
               │   Raw Documents     │
               │ (PDF / MD / Text)   │
               └─────────┬──────────┘
                         │
               [ Ingestion Pipeline ]
                         │
        ┌────────────────▼────────────────┐
        │  Chunking + Normalization        │
        │  (Unicode-safe, Indic-ready)     │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  Bi-Encoder Embedding            │
        │  (BGE-Small-en-v1.5, CPU)        │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  Local Vector Store              │
        │  (Qdrant, disk-backed)           │
        └────────────────┬────────────────┘
                         │
              ┌──────────▼──────────┐
              │     User Query       │
              └──────────┬──────────┘
                         │
        ┌────────────────▼────────────────┐
        │  Stage 1: Dense Retrieval        │
        │  (Top-K candidates)              │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  Stage 2: Cross-Encoder Rerank   │
        │  (BGE-reranker-base)             │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  Context Selection               │
        │  (Token-budget aware)            │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  LLM Generation (Groq)            │
        │  (Llama 3.3 70B)                  │
        └────────────────┬────────────────┘
                         │
                 ┌───────▼───────┐
                 │ Final Answer   │
                 └───────────────┘

4. Ingestion Pipeline (Index Construction)
4.1 Document Discovery

- Recursive directory traversal (data/raw_docs/)
- Supports: PDFs; Markdown; Plain text
- Implemented via SimpleDirectoryReader

Design rationale:
Directory-based ingestion mirrors real industrial knowledge bases (policy folders, manuals, reports).

4.2 Text Normalization & Chunking

Each document undergoes:
- Unicode normalization (critical for Indic scripts),
- whitespace and encoding cleanup,
- deterministic chunking.
- Chunking strategy:
- fixed-size, overlap-aware,
- optimized to balance:
- embedding fidelity,
- reranker effectiveness,
- LLM token limits.

4.3 Bi-Encoder Embedding (Stage 1 Backbone)

Model: BGE-Small-en-v1.5
Runtime: CPU
Memory footprint: ~133 MB

Why this model:
- Excellent semantic recall per MB,
- Stable embedding space,
- Proven industry adoption,
- Viable for low-VRAM machines.
- Each chunk → dense vector → metadata preserved.

4.4 Vector Persistence (Qdrant)

Deployment: Local Qdrant instance
Storage: Disk-backed, persistent
Data kept locally: embeddings + metadata only

Why Qdrant:
- Production-grade ANN search,
- Efficient filtering,
- Strong Python ecosystem,
- No cloud dependency.

5. Query Pipeline (RAG Execution)
5.1 Query Encoding

- User query embedded using same bi-encoder
- Ensures embedding space alignment
- Zero external calls at this stage

5.2 'Stage 1' — Semantic Retrieval (Recall)

- Qdrant performs ANN search
- Returns top-K candidate chunks
- Optimized for: speed; broad semantic coverage.

Important:
At this stage, precision is intentionally sacrificed for recall.

5.3 'Stage 2' — Neural Re-Ranking (Precision)

Model: BGE-Reranker-Base
Mechanism: Cross-encoder (query + passage jointly)

Scores each candidate pairwise

Why cross-encoder here:
- Deep token-level interaction,
- Strong semantic disambiguation,
- Removes “semantic noise” common in dense retrieval.
- Only the highest-scoring passages survive.

5.4 Context Assembly

Top reranked chunks selected
Token-budget aware trimming
Sources preserved for traceability

This ensures:
- maximal relevance per token,
- minimal hallucination surface.

5.5 Generation Layer (LLM)

Provider: Groq
Model: Llama 3.3 70B
Reason: ultra-low latency inference

Architectural decision:
- Retrieval & reasoning → local
- Generation → remote accelerator

This clean separation:
- preserves data sovereignty,
- avoids local GPU dependency,
- achieves near-real-time responses.

6. Hardware-Aware Design Strategy
Component	Execution
Chunking	CPU
Embedding	CPU / RAM
Reranking	CPU
Vector Search	Disk + RAM
Generation	Groq (remote)

Key insight:
Memory pressure is managed by moving intelligence into architecture, not hardware.

7. Failure Modes Addressed
Problem	Mitigation
Semantic drift	Cross-encoder reranking
Hallucination	Aggressive context filtering
VRAM exhaustion	CPU-only retrieval
Vendor lock-in	Modular model boundaries
Data leakage	Local-first embeddings

8. Extensibility Points

Vantage Core-RAG is intentionally modular:
Swap bi-encoder (e.g., domain-specific)
Add hybrid BM25 + dense retrieval


Plug in:
local LLM,
multi-index routing,
evaluation harness,
agentic query rewriting.
No architectural rewrite required.


9. Why This Is Not “Toy RAG”

- This system demonstrates:
- Industrial retrieval patterns
- Hardware-constrained engineering
- Clear separation of recall vs precision
- Production-grade vector infrastructure
- Explicit trade-off reasoning


It is designed as a core engine, not a demo notebook.
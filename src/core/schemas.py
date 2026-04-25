from dataclasses import dataclass
from typing import Optional


@dataclass
class Metrics:
    total_latency: float
    retrieval_time: float
    generation_time: float
    rerank_scores: list[float]
    tokens_used: int
    tokens_per_second: float
    peak_vram_mb: Optional[float] = None
    cache_hit: bool = False

@dataclass
class QueryResult:
    answer: str
    source_nodes: list
    metrics: Metrics
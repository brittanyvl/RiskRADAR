from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class SearchConfig:
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_index_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "search_data" / "bm25_index.pkl"
    )
    chunks_jsonl_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "extraction" / "json_data" / "chunks.jsonl"
    )
    default_model: str = "minilm"
    default_collection: str = "riskradar_minilm"
    rrf_k: int = 60
    semantic_weight: float = 0.6
    bm25_weight: float = 0.4
    default_limit: int = 20


SEARCH_CONFIG = SearchConfig()

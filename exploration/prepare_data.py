"""
exploration/prepare_data.py
---------------------------
Prepare and filter chunks for topic exploration.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import (
    CHUNKS_JSONL_PATH,
    EMBEDDINGS_DIR,
    EXPLORATION_DATA_DIR,
    BERTOPIC_CONFIG,
)

logger = logging.getLogger(__name__)


def load_chunks() -> pd.DataFrame:
    """Load all chunks from JSONL file."""
    logger.info(f"Loading chunks from {CHUNKS_JSONL_PATH}")

    chunks = []
    with open(CHUNKS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line)
                chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "report_id": chunk["report_id"],
                    "chunk_text": chunk["chunk_text"],
                    "section_name": chunk.get("section_name"),
                    "section_number": chunk.get("section_number"),
                    "token_count": chunk["token_count"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                })
            except Exception as e:
                logger.warning(f"Skipping line {line_num}: {e}")

    df = pd.DataFrame(chunks)
    logger.info(f"Loaded {len(df):,} chunks")
    return df


def filter_chunks_by_section(
    df: pd.DataFrame,
    filter_sections: list[str] | None = None,
    min_tokens: int | None = None,
) -> pd.DataFrame:
    """Filter chunks to specified sections."""
    if filter_sections is None:
        filter_sections = BERTOPIC_CONFIG.filter_sections
    if min_tokens is None:
        min_tokens = BERTOPIC_CONFIG.min_chunk_tokens

    original_count = len(df)
    pattern = "|".join(filter_sections)

    section_mask = df["section_name"].str.upper().str.contains(
        pattern, na=False, regex=True, case=False
    )
    token_mask = df["token_count"] >= min_tokens
    filtered = df[section_mask & token_mask].copy()

    logger.info(
        f"Filtered {original_count:,} -> {len(filtered):,} chunks "
        f"({len(filtered)/original_count*100:.1f}%) "
        f"(sections: {filter_sections}, min_tokens: {min_tokens})"
    )

    section_counts = filtered["section_name"].value_counts().head(10)
    logger.info(f"Top sections in filtered data:\n{section_counts}")

    return filtered


def load_mika_embeddings(chunk_ids: list[str] | None = None) -> pd.DataFrame:
    """Load pre-computed MIKA embeddings."""
    embeddings_path = EMBEDDINGS_DIR / "mika_embeddings.parquet"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"MIKA embeddings not found at {embeddings_path}. "
            "Run `python -m embeddings.cli embed mika` first."
        )

    logger.info(f"Loading MIKA embeddings from {embeddings_path}")

    table = pq.read_table(embeddings_path)
    df = table.to_pandas()

    if chunk_ids is not None:
        chunk_id_set = set(chunk_ids)
        df_filtered = df[df["chunk_id"].isin(chunk_id_set)].copy()
        logger.info(
            f"Filtered to {len(df_filtered):,} embeddings "
            f"({len(df_filtered)/len(chunk_ids)*100:.1f}% of requested chunks)"
        )
        return df_filtered

    logger.info(f"Loaded {len(df):,} embeddings")
    return df


def prepare_discovery_dataset(
    filter_sections: list[str] | None = None,
    min_tokens: int | None = None,
    save_filtered: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Full pipeline: Load chunks, filter, and load embeddings."""
    chunks = load_chunks()
    filtered_chunks = filter_chunks_by_section(
        chunks,
        filter_sections=filter_sections,
        min_tokens=min_tokens,
    )

    chunk_ids = filtered_chunks["chunk_id"].tolist()
    embeddings_df = load_mika_embeddings(chunk_ids)

    # Only keep chunk_id and embedding columns to avoid duplicates
    embeddings_df = embeddings_df[["chunk_id", "embedding"]]

    merged = filtered_chunks.merge(embeddings_df, on="chunk_id", how="inner")
    logger.info(f"Merged data: {len(merged):,} chunks with embeddings")

    documents_df = merged.drop(columns=["embedding"])
    embeddings_array = np.vstack(merged["embedding"].values)

    logger.info(f"Embeddings shape: {embeddings_array.shape}")

    if save_filtered:
        output_path = EXPLORATION_DATA_DIR / "filtered_chunks.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in documents_df.iterrows():
                f.write(json.dumps(row.to_dict(), default=str) + "\n")
        logger.info(f"Saved filtered chunks to {output_path}")

    return documents_df, embeddings_array


def get_section_statistics(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Get statistics about sections in the chunks."""
    if df is None:
        df = load_chunks()

    stats = df.groupby("section_name").agg({
        "chunk_id": "count",
        "token_count": ["mean", "sum"],
        "report_id": "nunique",
    }).round(1)

    stats.columns = ["chunk_count", "avg_tokens", "total_tokens", "report_count"]
    stats = stats.sort_values("chunk_count", ascending=False)

    return stats

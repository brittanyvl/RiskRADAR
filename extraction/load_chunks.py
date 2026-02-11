"""
extraction/load_chunks.py
-------------------------
Load chunk metadata from chunks.jsonl into SQLite chunks table.
Does NOT store chunk_text (kept in JSONL only per schema design).

Usage:
    python -m extraction.load_chunks
    python -m risk_profiler.cli load-chunks
"""

import json
import sqlite3
from pathlib import Path

from riskradar.config import DB_PATH, PROJECT_ROOT

CHUNKS_JSONL = PROJECT_ROOT / "extraction" / "json_data" / "chunks.jsonl"
BATCH_SIZE = 500


def load_chunks(conn: sqlite3.Connection, chunks_path: Path = None) -> dict:
    """
    Load chunk metadata from JSONL into the SQLite chunks table.

    Args:
        conn: SQLite connection
        chunks_path: Path to chunks.jsonl (uses default if None)

    Returns:
        Dict with load summary stats
    """
    chunks_path = chunks_path or CHUNKS_JSONL

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks JSONL not found: {chunks_path}")

    print(f"Loading chunks from {chunks_path}")

    rows = []
    section_counts = {}
    report_ids = set()

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)

            report_id = chunk["report_id"]
            report_ids.add(report_id)

            section = chunk.get("section_name", "")
            section_counts[section] = section_counts.get(section, 0) + 1

            rows.append((
                chunk["chunk_id"],
                report_id,
                chunk["chunk_sequence"],
                chunk["page_start"],
                chunk["page_end"],
                json.dumps(chunk.get("page_list")) if chunk.get("page_list") is not None else None,
                chunk["char_start"],
                chunk["char_end"],
                chunk.get("section_name"),
                chunk.get("section_number"),
                chunk.get("section_detection_method"),
                chunk["token_count"],
                chunk.get("overlap_tokens", 0),
                chunk["text_source"],
                json.dumps(chunk.get("page_sources")) if chunk.get("page_sources") is not None else None,
                json.dumps(chunk.get("source_quality")) if chunk.get("source_quality") is not None else None,
                1 if chunk.get("has_footnotes") else 0,
                json.dumps(chunk.get("footnotes")) if chunk.get("footnotes") is not None else None,
                json.dumps(chunk.get("quality_flags")) if chunk.get("quality_flags") is not None else None,
                "extraction/json_data/chunks.jsonl",
                chunk.get("pipeline_version"),
                chunk.get("created_at"),
            ))

    # Batch insert
    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        conn.executemany(
            """INSERT OR IGNORE INTO chunks
               (chunk_id, report_id, chunk_sequence,
                page_start, page_end, page_list_json,
                char_start, char_end,
                section_name, section_number, section_detection_method,
                token_count, overlap_tokens,
                text_source, page_sources_json, source_quality_json,
                has_footnotes, footnotes_json,
                quality_flags_json,
                jsonl_path, pipeline_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            batch,
        )
        inserted += len(batch)
    conn.commit()

    # Summary
    print("\n" + "=" * 60)
    print("CHUNK LOAD COMPLETE")
    print("=" * 60)
    print(f"Total chunks loaded: {len(rows):,}")
    print(f"Reports covered: {len(report_ids)}")
    print(f"\nSection distribution (top 15):")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {section or '(none)':40s} {count:>6,}")
    print("=" * 60)

    return {
        "total_loaded": len(rows),
        "reports_covered": len(report_ids),
        "section_counts": section_counts,
    }


if __name__ == "__main__":
    conn = sqlite3.connect(str(DB_PATH))
    load_chunks(conn)
    conn.close()

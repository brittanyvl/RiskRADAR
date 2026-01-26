"""
taxonomy/hierarchical_mapper.py
-------------------------------
Level 2 subcategory classification building on existing L1 results.

This module extends the existing CICTT L1 classification (from mapper.py)
with industry-standard L2 subcategories. It does NOT re-run L1 - it
loads existing L1 results and adds subcategory classification.

Workflow:
1. Load existing L1 chunk assignments (from mapper.py run)
2. Load MIKA embeddings for chunks with L2-enabled L1 categories
3. Classify those chunks to L2 subcategories
4. Aggregate to report level

References:
- CICTT Aviation Occurrence Categories v4.7
- IATA LOC-I Analysis Framework
- HFACS (Shappell & Wiegmann, 2000)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from .cictt import CICTT_CATEGORIES, CICTT_BY_CODE, get_primary_categories
from .subcategories import (
    ALL_SUBCATEGORIES,
    SUBCATEGORY_BY_CODE,
    PARENT_TO_SUBCATEGORIES,
    HFACS_SUBCATEGORIES,
    get_subcategories_for_parent,
    has_subcategories,
)
from .config import (
    CHUNKS_JSONL_PATH,
    EMBEDDINGS_DIR,
    TAXONOMY_DATA_DIR,
    TAXONOMY_CONFIG,
)

logger = logging.getLogger(__name__)


class HierarchicalMapper:
    """
    Two-pass hierarchical mapper for CICTT + subcategory classification.

    Implements the embedding-based approach used for L1 classification,
    extended to support hierarchical L2 subcategories.
    """

    def __init__(self, config=None):
        """
        Initialize the hierarchical mapper.

        Args:
            config: TaxonomyConfig instance (uses default if None)
        """
        self.config = config or TAXONOMY_CONFIG
        self.model = None  # Lazy-loaded

        # Embeddings (computed once)
        self.l1_embeddings: dict[str, np.ndarray] = {}
        self.l2_embeddings: dict[str, dict[str, np.ndarray]] = {}

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
        return self.model

    def compute_l1_embeddings(self) -> dict[str, np.ndarray]:
        """
        Compute embeddings for all L1 CICTT categories.

        Returns:
            Dict mapping category code to averaged embedding vector
        """
        if self.l1_embeddings:
            return self.l1_embeddings

        model = self._load_model()
        logger.info("Computing L1 CICTT category embeddings...")

        for cat in get_primary_categories():
            # Combine description and seed phrases
            texts = [cat.description] + cat.seed_phrases
            embeddings = model.encode(texts, convert_to_numpy=True)
            self.l1_embeddings[cat.code] = np.mean(embeddings, axis=0)

        logger.info(f"Computed embeddings for {len(self.l1_embeddings)} L1 categories")
        return self.l1_embeddings

    def compute_l2_embeddings(self) -> dict[str, dict[str, np.ndarray]]:
        """
        Compute embeddings for all L2 subcategories.

        Returns:
            Nested dict: {parent_code: {subcategory_code: embedding}}
        """
        if self.l2_embeddings:
            return self.l2_embeddings

        model = self._load_model()
        logger.info("Computing L2 subcategory embeddings...")

        for parent_code in self.config.l2_enabled_categories:
            subcategories = get_subcategories_for_parent(parent_code)
            if not subcategories:
                continue

            self.l2_embeddings[parent_code] = {}

            for subcat in subcategories:
                texts = [subcat.description] + subcat.seed_phrases
                embeddings = model.encode(texts, convert_to_numpy=True)
                self.l2_embeddings[parent_code][subcat.code] = np.mean(embeddings, axis=0)

        # Also compute HFACS embeddings (applied across applicable categories)
        self.l2_embeddings["HFACS"] = {}
        for hfacs in HFACS_SUBCATEGORIES:
            texts = [hfacs.description] + hfacs.seed_phrases
            embeddings = model.encode(texts, convert_to_numpy=True)
            self.l2_embeddings["HFACS"][hfacs.code] = np.mean(embeddings, axis=0)

        total_l2 = sum(len(subs) for subs in self.l2_embeddings.values())
        logger.info(f"Computed embeddings for {total_l2} L2 subcategories")
        return self.l2_embeddings

    def load_chunk_embeddings(self, chunk_ids: list[str]) -> pd.DataFrame:
        """Load pre-computed MIKA embeddings for specified chunks."""
        embeddings_path = EMBEDDINGS_DIR / "mika_embeddings.parquet"

        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"MIKA embeddings not found at {embeddings_path}. "
                "Run `python -m embeddings.cli embed mika` first."
            )

        logger.info(f"Loading MIKA embeddings from {embeddings_path}")
        table = pq.read_table(embeddings_path)
        df = table.to_pandas()

        chunk_id_set = set(chunk_ids)
        df_filtered = df[df["chunk_id"].isin(chunk_id_set)].copy()

        logger.info(f"Loaded {len(df_filtered):,} embeddings for {len(chunk_ids):,} chunks")
        return df_filtered[["chunk_id", "embedding"]]

    def load_existing_l1_results(self, l1_run_id: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load existing L1 classification results from mapper.py run.

        Args:
            l1_run_id: The run ID of the existing L1 classification

        Returns:
            Tuple of (chunk_assignments, report_categories) DataFrames
        """
        # These files are created by mapper.py
        chunk_path = TAXONOMY_DATA_DIR / f"chunk_assignments_run{l1_run_id}.parquet"
        report_path = TAXONOMY_DATA_DIR / f"report_categories_run{l1_run_id}.parquet"

        if not chunk_path.exists():
            raise FileNotFoundError(
                f"L1 chunk assignments not found at {chunk_path}. "
                f"Run `python -m taxonomy.cli map --run-id {l1_run_id}` first."
            )

        if not report_path.exists():
            raise FileNotFoundError(
                f"L1 report categories not found at {report_path}. "
                f"Run `python -m taxonomy.cli map --run-id {l1_run_id}` first."
            )

        chunk_assignments = pd.read_parquet(chunk_path)
        report_categories = pd.read_parquet(report_path)

        logger.info(f"Loaded existing L1 results from run {l1_run_id}:")
        logger.info(f"  - {len(chunk_assignments):,} chunk assignments")
        logger.info(f"  - {len(report_categories):,} report-category assignments")
        logger.info(f"  - {chunk_assignments['report_id'].nunique()} unique reports")

        return chunk_assignments, report_categories

    def classify_l1(
        self,
        chunk_embeddings: np.ndarray,
        chunk_ids: list[str],
        report_ids: list[str],
    ) -> pd.DataFrame:
        """
        Classify chunks to L1 CICTT categories.

        Args:
            chunk_embeddings: Matrix of chunk embeddings (n_chunks, dim)
            chunk_ids: List of chunk IDs
            report_ids: List of report IDs (parallel to chunk_ids)

        Returns:
            DataFrame with columns: chunk_id, report_id, category_code, similarity, rank
        """
        l1_embeddings = self.compute_l1_embeddings()
        codes = list(l1_embeddings.keys())
        cat_matrix = np.vstack([l1_embeddings[code] for code in codes])

        # Normalize for cosine similarity
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        cat_norms = np.linalg.norm(cat_matrix, axis=1, keepdims=True)

        chunk_normalized = chunk_embeddings / (chunk_norms + 1e-8)
        cat_normalized = cat_matrix / (cat_norms + 1e-8)

        # Compute similarities: (n_chunks, n_categories)
        similarities = chunk_normalized @ cat_normalized.T

        logger.info(f"Computed L1 similarity matrix: {similarities.shape}")

        # Assign categories
        assignments = []
        threshold = self.config.min_similarity_threshold
        top_k = self.config.top_k_categories

        for idx in range(len(chunk_ids)):
            chunk_sims = similarities[idx]
            top_indices = np.argsort(chunk_sims)[::-1][:top_k]

            for rank, cat_idx in enumerate(top_indices):
                sim = chunk_sims[cat_idx]
                if sim >= threshold:
                    assignments.append({
                        "chunk_id": chunk_ids[idx],
                        "report_id": report_ids[idx],
                        "category_code": codes[cat_idx],
                        "similarity": float(sim),
                        "rank": rank + 1,
                    })

        assignments_df = pd.DataFrame(assignments)
        logger.info(f"Generated {len(assignments_df):,} L1 chunk assignments")

        return assignments_df

    def classify_l2(
        self,
        chunk_embeddings: np.ndarray,
        chunk_ids: list[str],
        report_ids: list[str],
        l1_assignments: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Classify chunks to L2 subcategories based on L1 assignments.

        Only processes chunks that have L1 assignments with L2-enabled categories.

        Args:
            chunk_embeddings: Matrix of chunk embeddings (n_chunks, dim)
            chunk_ids: List of chunk IDs
            report_ids: List of report IDs
            l1_assignments: L1 chunk assignments DataFrame

        Returns:
            DataFrame with L2 assignments
        """
        if not self.config.enable_l2_classification:
            logger.info("L2 classification disabled, skipping")
            return pd.DataFrame()

        l2_embeddings = self.compute_l2_embeddings()

        # Build chunk_id to embedding index mapping
        chunk_id_to_idx = {cid: idx for idx, cid in enumerate(chunk_ids)}

        # Filter L1 assignments to only L2-enabled categories
        l2_enabled = set(self.config.l2_enabled_categories)
        eligible_l1 = l1_assignments[l1_assignments["category_code"].isin(l2_enabled)]

        logger.info(
            f"Processing L2 classification for {len(eligible_l1)} L1 assignments "
            f"from {eligible_l1['chunk_id'].nunique()} unique chunks"
        )

        assignments = []
        threshold = self.config.min_l2_similarity_threshold
        top_k = self.config.top_k_subcategories
        min_combined = self.config.min_combined_confidence

        for _, row in eligible_l1.iterrows():
            chunk_id = row["chunk_id"]
            report_id = row["report_id"]
            parent_code = row["category_code"]
            l1_similarity = row["similarity"]

            # Get chunk embedding
            if chunk_id not in chunk_id_to_idx:
                continue
            chunk_embedding = chunk_embeddings[chunk_id_to_idx[chunk_id]]

            # Get subcategory embeddings for this parent
            if parent_code not in l2_embeddings:
                continue

            subcats = l2_embeddings[parent_code]
            subcat_codes = list(subcats.keys())
            subcat_matrix = np.vstack([subcats[code] for code in subcat_codes])

            # Normalize and compute similarities
            chunk_norm = np.linalg.norm(chunk_embedding)
            subcat_norms = np.linalg.norm(subcat_matrix, axis=1, keepdims=True)

            chunk_normalized = chunk_embedding / (chunk_norm + 1e-8)
            subcat_normalized = subcat_matrix / (subcat_norms + 1e-8)

            sims = chunk_normalized @ subcat_normalized.T

            # Get top-k subcategories
            top_indices = np.argsort(sims)[::-1][:top_k]

            for rank, sub_idx in enumerate(top_indices):
                l2_sim = sims[sub_idx]
                combined = l1_similarity * l2_sim

                if l2_sim >= threshold and combined >= min_combined:
                    assignments.append({
                        "chunk_id": chunk_id,
                        "report_id": report_id,
                        "parent_code": parent_code,
                        "subcategory_code": subcat_codes[sub_idx],
                        "l2_similarity": float(l2_sim),
                        "combined_confidence": float(combined),
                        "rank": rank + 1,
                    })

            # Also check HFACS subcategories if applicable
            if parent_code in self.config.hfacs_applicable_categories:
                hfacs_subs = l2_embeddings.get("HFACS", {})
                if hfacs_subs:
                    hfacs_codes = list(hfacs_subs.keys())
                    hfacs_matrix = np.vstack([hfacs_subs[c] for c in hfacs_codes])
                    hfacs_norms = np.linalg.norm(hfacs_matrix, axis=1, keepdims=True)
                    hfacs_normalized = hfacs_matrix / (hfacs_norms + 1e-8)

                    hfacs_sims = chunk_normalized @ hfacs_normalized.T
                    top_hfacs = np.argsort(hfacs_sims)[::-1][:1]  # Top 1 HFACS

                    for h_idx in top_hfacs:
                        h_sim = hfacs_sims[h_idx]
                        h_combined = l1_similarity * h_sim

                        if h_sim >= threshold and h_combined >= min_combined:
                            assignments.append({
                                "chunk_id": chunk_id,
                                "report_id": report_id,
                                "parent_code": parent_code,
                                "subcategory_code": hfacs_codes[h_idx],
                                "l2_similarity": float(h_sim),
                                "combined_confidence": float(h_combined),
                                "rank": 99,  # HFACS as supplementary
                            })

        assignments_df = pd.DataFrame(assignments)
        logger.info(f"Generated {len(assignments_df):,} L2 chunk assignments")

        return assignments_df

    def aggregate_l1_to_report(self, l1_chunk_assignments: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate chunk-level L1 assignments to report-level categories.

        Args:
            l1_chunk_assignments: DataFrame from classify_l1

        Returns:
            DataFrame with report-level L1 categories
        """
        if l1_chunk_assignments.empty:
            return pd.DataFrame()

        # Group by report and category
        report_cats = l1_chunk_assignments.groupby(["report_id", "category_code"]).agg({
            "similarity": ["mean", "max", "count"],
            "chunk_id": "count"
        }).reset_index()

        report_cats.columns = [
            "report_id", "category_code",
            "avg_similarity", "max_similarity", "n_assignments", "n_chunks"
        ]

        # Compute weighted score
        report_cats["score"] = (
            0.4 * report_cats["avg_similarity"] +
            0.4 * report_cats["max_similarity"] +
            0.2 * np.minimum(report_cats["n_chunks"] / 5, 1.0)
        )

        # Rank within each report
        report_cats["rank"] = report_cats.groupby("report_id")["score"].rank(
            ascending=False, method="first"
        ).astype(int)

        # Keep top categories
        max_cats = self.config.max_categories_per_report
        report_cats = report_cats[report_cats["rank"] <= max_cats].copy()

        # Normalize to percentages
        report_totals = report_cats.groupby("report_id")["score"].sum()
        report_cats["pct_contribution"] = report_cats.apply(
            lambda row: 100 * row["score"] / report_totals[row["report_id"]], axis=1
        ).round(1)

        # Add category names
        report_cats["category_name"] = report_cats["category_code"].map(
            lambda c: CICTT_BY_CODE[c].name if c in CICTT_BY_CODE else c
        )

        logger.info(f"Aggregated to {len(report_cats):,} report-level L1 assignments")

        return report_cats

    def aggregate_l2_to_report(self, l2_chunk_assignments: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate chunk-level L2 assignments to report-level subcategories.

        Args:
            l2_chunk_assignments: DataFrame from classify_l2

        Returns:
            DataFrame with report-level L2 subcategories
        """
        if l2_chunk_assignments.empty:
            return pd.DataFrame()

        # Group by report, parent, and subcategory
        report_subs = l2_chunk_assignments.groupby(
            ["report_id", "parent_code", "subcategory_code"]
        ).agg({
            "l2_similarity": ["mean", "max"],
            "combined_confidence": ["mean", "max"],
            "chunk_id": "count"
        }).reset_index()

        report_subs.columns = [
            "report_id", "parent_code", "subcategory_code",
            "avg_similarity", "max_similarity",
            "avg_confidence", "max_confidence",
            "n_chunks"
        ]

        # Compute score
        report_subs["score"] = (
            0.3 * report_subs["avg_similarity"] +
            0.3 * report_subs["max_similarity"] +
            0.2 * report_subs["avg_confidence"] +
            0.2 * np.minimum(report_subs["n_chunks"] / 3, 1.0)
        )

        # Rank within each report-parent combination
        report_subs["rank"] = report_subs.groupby(
            ["report_id", "parent_code"]
        )["score"].rank(ascending=False, method="first").astype(int)

        # Keep top subcategories per parent
        max_subs = self.config.max_subcategories_per_parent
        report_subs = report_subs[report_subs["rank"] <= max_subs].copy()

        # Normalize to percentages within parent
        parent_totals = report_subs.groupby(
            ["report_id", "parent_code"]
        )["score"].sum()
        report_subs["pct_of_parent"] = report_subs.apply(
            lambda row: 100 * row["score"] / parent_totals[(row["report_id"], row["parent_code"])],
            axis=1
        ).round(1)

        # Combined confidence (average of the two confidence values)
        report_subs["combined_confidence"] = (
            report_subs["avg_confidence"] + report_subs["max_confidence"]
        ) / 2

        # Add subcategory names
        report_subs["subcategory_name"] = report_subs["subcategory_code"].map(
            lambda c: SUBCATEGORY_BY_CODE[c].name if c in SUBCATEGORY_BY_CODE else c
        )

        logger.info(f"Aggregated to {len(report_subs):,} report-level L2 assignments")

        return report_subs

    def save_results(
        self,
        l1_chunks: pd.DataFrame,
        l1_reports: pd.DataFrame,
        l2_chunks: pd.DataFrame,
        l2_reports: pd.DataFrame,
        run_id: int = 1,
    ) -> dict:
        """
        Save classification results to parquet files.

        Returns:
            Dict with file paths
        """
        paths = {}

        # L1 chunk assignments
        if not l1_chunks.empty:
            path = TAXONOMY_DATA_DIR / f"chunk_l1_run{run_id}.parquet"
            l1_chunks.to_parquet(path, index=False)
            paths["l1_chunks"] = path

        # L1 report assignments
        if not l1_reports.empty:
            path = TAXONOMY_DATA_DIR / f"report_l1_run{run_id}.parquet"
            l1_reports.to_parquet(path, index=False)
            paths["l1_reports"] = path

        # L2 chunk assignments
        if not l2_chunks.empty:
            path = TAXONOMY_DATA_DIR / f"chunk_l2_run{run_id}.parquet"
            l2_chunks.to_parquet(path, index=False)
            paths["l2_chunks"] = path

        # L2 report assignments
        if not l2_reports.empty:
            path = TAXONOMY_DATA_DIR / f"report_l2_run{run_id}.parquet"
            l2_reports.to_parquet(path, index=False)
            paths["l2_reports"] = path

        # Summary statistics
        stats = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config_version": self.config.version,
            "l1_chunk_assignments": len(l1_chunks),
            "l1_report_assignments": len(l1_reports),
            "l1_categories_used": l1_reports["category_code"].nunique() if not l1_reports.empty else 0,
            "l1_reports_classified": l1_reports["report_id"].nunique() if not l1_reports.empty else 0,
            "l2_enabled": self.config.enable_l2_classification,
            "l2_chunk_assignments": len(l2_chunks),
            "l2_report_assignments": len(l2_reports),
            "l2_subcategories_used": l2_reports["subcategory_code"].nunique() if not l2_reports.empty else 0,
        }

        stats_path = TAXONOMY_DATA_DIR / f"hierarchical_stats_run{run_id}.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        paths["stats"] = stats_path

        logger.info(f"Saved results to {TAXONOMY_DATA_DIR}")

        return {"paths": paths, "stats": stats}


def load_hierarchical_results(run_id: int = 1) -> dict:
    """Load previous hierarchical classification results."""
    results = {}

    # L1 chunks
    path = TAXONOMY_DATA_DIR / f"chunk_l1_run{run_id}.parquet"
    if path.exists():
        results["l1_chunks"] = pd.read_parquet(path)

    # L1 reports
    path = TAXONOMY_DATA_DIR / f"report_l1_run{run_id}.parquet"
    if path.exists():
        results["l1_reports"] = pd.read_parquet(path)

    # L2 chunks
    path = TAXONOMY_DATA_DIR / f"chunk_l2_run{run_id}.parquet"
    if path.exists():
        results["l2_chunks"] = pd.read_parquet(path)

    # L2 reports
    path = TAXONOMY_DATA_DIR / f"report_l2_run{run_id}.parquet"
    if path.exists():
        results["l2_reports"] = pd.read_parquet(path)

    # Stats
    path = TAXONOMY_DATA_DIR / f"hierarchical_stats_run{run_id}.json"
    if path.exists():
        with open(path) as f:
            results["stats"] = json.load(f)

    return results

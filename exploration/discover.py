"""
exploration/discover.py
-----------------------
BERTopic-based topic discovery for freeform exploration.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BERTOPIC_CONFIG, EXPLORATION_DATA_DIR, EXPLORATION_VIZ_DIR
from .prepare_data import prepare_discovery_dataset

logger = logging.getLogger(__name__)


def create_bertopic_model():
    """Create BERTopic model with configuration."""
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    config = BERTOPIC_CONFIG

    umap_model = UMAP(
        n_neighbors=config.umap_n_neighbors,
        n_components=config.umap_n_components,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.umap_random_state,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric=config.hdbscan_metric,
        cluster_selection_method=config.hdbscan_cluster_selection_method,
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        ngram_range=config.n_gram_range,
        stop_words="english",
        min_df=2,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=config.top_n_words,
        min_topic_size=config.min_topic_size,
        nr_topics=config.nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    logger.info("Created BERTopic model")
    return topic_model


def fit_topic_model(
    documents_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple:
    """Fit BERTopic model to chunks using pre-computed embeddings."""
    logger.info("Fitting BERTopic model...")

    documents = documents_df["chunk_text"].tolist()
    topic_model = create_bertopic_model()

    start_time = datetime.now()
    topics, probs = topic_model.fit_transform(documents, embeddings)
    elapsed = (datetime.now() - start_time).total_seconds()

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = (np.array(topics) == -1).sum()

    logger.info(
        f"Topic modeling complete in {elapsed:.1f}s. "
        f"Discovered {n_topics} topics, {n_outliers} outlier chunks "
        f"({n_outliers/len(topics)*100:.1f}%)"
    )

    return topic_model, topics, probs, topic_info


def create_topic_assignments(
    documents_df: pd.DataFrame,
    topics: list[int],
    probs: np.ndarray,
) -> pd.DataFrame:
    """Create DataFrame of topic assignments per chunk."""
    max_probs = [p.max() if len(p) > 0 else 0.0 for p in probs]

    assignments = pd.DataFrame({
        "chunk_id": documents_df["chunk_id"].values,
        "report_id": documents_df["report_id"].values,
        "section_name": documents_df["section_name"].values,
        "topic_id": topics,
        "topic_probability": max_probs,
    })

    return assignments


def save_discovery_results(
    topic_model,
    documents_df: pd.DataFrame,
    topics: list[int],
    probs: np.ndarray,
    topic_info: pd.DataFrame,
    run_id: int = 1,
) -> dict:
    """Save all discovery results to disk."""
    paths = {}

    model_path = EXPLORATION_DATA_DIR / f"topic_model_run{run_id}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(topic_model, f)
    paths["model"] = model_path
    logger.info(f"Saved topic model to {model_path}")

    assignments = create_topic_assignments(documents_df, topics, probs)
    assignments_path = EXPLORATION_DATA_DIR / f"topic_assignments_run{run_id}.parquet"
    assignments.to_parquet(assignments_path, index=False)
    paths["assignments"] = assignments_path
    logger.info(f"Saved topic assignments to {assignments_path}")

    info_path = EXPLORATION_DATA_DIR / f"topic_info_run{run_id}.parquet"
    topic_info.to_parquet(info_path, index=False)
    paths["topic_info"] = info_path
    logger.info(f"Saved topic info to {info_path}")

    topics_json = []
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        topic_words = topic_model.get_topic(row["Topic"])
        topics_json.append({
            "topic_id": int(row["Topic"]),
            "name": row["Name"],
            "count": int(row["Count"]),
            "keywords": [{"word": w, "score": float(s)} for w, s in topic_words[:10]],
        })

    keywords_path = EXPLORATION_DATA_DIR / f"topic_keywords_run{run_id}.json"
    with open(keywords_path, "w", encoding="utf-8") as f:
        json.dump(topics_json, f, indent=2)
    paths["keywords"] = keywords_path
    logger.info(f"Saved topic keywords to {keywords_path}")

    return paths


def generate_visualizations(
    topic_model,
    documents_df: pd.DataFrame,
    embeddings: np.ndarray,
    run_id: int = 1,
) -> dict:
    """Generate BERTopic visualizations."""
    paths = {}
    documents = documents_df["chunk_text"].tolist()

    try:
        fig = topic_model.visualize_topics()
        path = EXPLORATION_VIZ_DIR / f"intertopic_distance_run{run_id}.html"
        fig.write_html(str(path))
        paths["intertopic_distance"] = path
        logger.info(f"Saved intertopic distance map to {path}")
    except Exception as e:
        logger.warning(f"Could not generate intertopic distance map: {e}")

    try:
        fig = topic_model.visualize_hierarchy()
        path = EXPLORATION_VIZ_DIR / f"topic_hierarchy_run{run_id}.html"
        fig.write_html(str(path))
        paths["hierarchy"] = path
        logger.info(f"Saved topic hierarchy to {path}")
    except Exception as e:
        logger.warning(f"Could not generate topic hierarchy: {e}")

    try:
        fig = topic_model.visualize_barchart(top_n_topics=20)
        path = EXPLORATION_VIZ_DIR / f"topic_barchart_run{run_id}.html"
        fig.write_html(str(path))
        paths["barchart"] = path
        logger.info(f"Saved topic barchart to {path}")
    except Exception as e:
        logger.warning(f"Could not generate topic barchart: {e}")

    try:
        fig = topic_model.visualize_heatmap(n_clusters=min(20, len(topic_model.get_topics())))
        path = EXPLORATION_VIZ_DIR / f"topic_heatmap_run{run_id}.html"
        fig.write_html(str(path))
        paths["heatmap"] = path
        logger.info(f"Saved topic heatmap to {path}")
    except Exception as e:
        logger.warning(f"Could not generate topic heatmap: {e}")

    return paths


def discover_topics(
    filter_sections: list[str] | None = None,
    min_tokens: int | None = None,
    run_id: int = 1,
    generate_viz: bool = True,
) -> dict:
    """Full topic discovery pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Topic Exploration")
    logger.info("=" * 60)

    logger.info("Step 1: Preparing discovery dataset...")
    documents_df, embeddings = prepare_discovery_dataset(
        filter_sections=filter_sections,
        min_tokens=min_tokens,
    )

    logger.info("Step 2: Fitting BERTopic model...")
    topic_model, topics, probs, topic_info = fit_topic_model(
        documents_df, embeddings
    )

    logger.info("Step 3: Saving discovery results...")
    paths = save_discovery_results(
        topic_model, documents_df, topics, probs, topic_info, run_id
    )

    if generate_viz:
        logger.info("Step 4: Generating visualizations...")
        viz_paths = generate_visualizations(
            topic_model, documents_df, embeddings, run_id
        )
        paths.update(viz_paths)

    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = (np.array(topics) == -1).sum()
    outlier_pct = n_outliers / len(topics) * 100

    results = {
        "run_id": run_id,
        "n_chunks": len(documents_df),
        "n_topics": n_topics,
        "n_outliers": n_outliers,
        "outlier_pct": outlier_pct,
        "paths": paths,
        "topic_info": topic_info,
    }

    logger.info("=" * 60)
    logger.info("Topic Exploration Complete!")
    logger.info(f"  Chunks processed: {len(documents_df):,}")
    logger.info(f"  Topics discovered: {n_topics}")
    logger.info(f"  Outlier chunks: {n_outliers:,} ({outlier_pct:.1f}%)")
    logger.info("=" * 60)

    return results


def load_topic_model(run_id: int = 1):
    """Load a previously saved topic model."""
    model_path = EXPLORATION_DATA_DIR / f"topic_model_run{run_id}.pkl"
    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)
    logger.info(f"Loaded topic model from {model_path}")
    return topic_model


def load_topic_assignments(run_id: int = 1) -> pd.DataFrame:
    """Load topic assignments from a previous run."""
    path = EXPLORATION_DATA_DIR / f"topic_assignments_run{run_id}.parquet"
    return pd.read_parquet(path)


def load_topic_info(run_id: int = 1) -> pd.DataFrame:
    """Load topic info from a previous run."""
    path = EXPLORATION_DATA_DIR / f"topic_info_run{run_id}.parquet"
    return pd.read_parquet(path)

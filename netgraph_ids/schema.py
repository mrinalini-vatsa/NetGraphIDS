"""
Single source of truth for NetGraph-IDS data schemas.

All pipeline stages must import column names and validators from this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from torch_geometric.data import Data

SCHEMA_VERSION = 1

# Canonical column names used throughout the pipeline after preprocessing.
ID_COLUMNS: tuple[str, ...] = (
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "timestamp",
)
LABEL_COLUMNS: tuple[str, ...] = ("label", "attack", "attack_type")

# Raw CICIDS GeneratedLabelledFlows headers (before normalize_columns).
RAW_IP_HEADERS: tuple[str, ...] = ("Source IP", "Destination IP")
RAW_LABEL_HEADER = "Label"

# Maps normalized intermediate names → canonical ID column names.
COLUMN_ALIASES: dict[str, str] = {
    "source_ip": "src_ip",
    "destination_ip": "dst_ip",
    "source_port": "src_port",
    "destination_port": "dst_port",
}

# Columns excluded from ML feature extraction.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        *ID_COLUMNS,
        *LABEL_COLUMNS,
        "flow_id",
    }
)

# Graph node features = mean(flow features) + in_degree + out_degree
GRAPH_DEGREE_FEATURES = 2


def required_graph_columns() -> tuple[str, ...]:
    """Minimum columns required to build IP graphs."""
    return ("src_ip", "dst_ip", "attack")


def node_feature_dim(num_flow_features: int) -> int:
    return num_flow_features + GRAPH_DEGREE_FEATURES


def validate_raw_csv_header(header_line: str, filename: str) -> None:
    """Validate a raw CSV header has IPs and labels (GeneratedLabelledFlows format)."""
    if not header_line or header_line.lstrip().startswith("<!DOCTYPE"):
        raise ValueError(
            f"{filename} is not a valid CSV (HTML or empty response). "
            "Use GeneratedLabelledFlows files with Source IP and Destination IP."
        )
    header_lower = header_line.lower()
    has_ip = "source ip" in header_lower and "destination ip" in header_lower
    has_label = "label" in header_lower
    if not has_ip:
        raise ValueError(
            f"{filename} is missing Source IP / Destination IP columns. "
            "MachineLearningCSV files cannot be used for graph-based detection. "
            "Re-download with: netgraph-ids download --force"
        )
    if not has_label:
        raise ValueError(f"{filename} is missing a Label column.")


def validate_flow_dataframe(
    df: pd.DataFrame,
    stage: str,
    require_labels: bool = True,
    feat_cols: Optional[list[str]] = None,
) -> None:
    """Verify a flows DataFrame matches the canonical processed schema."""
    missing_graph = [col for col in required_graph_columns() if col not in df.columns]
    if missing_graph:
        raise ValueError(
            f"[{stage}] flows data is missing required columns: {missing_graph}. "
            f"Available columns: {list(df.columns)[:20]}… "
            "Re-run preprocessing after downloading GeneratedLabelledFlows data."
        )

    if require_labels:
        missing_labels = [col for col in LABEL_COLUMNS if col not in df.columns]
        if missing_labels:
            raise ValueError(
                f"[{stage}] flows data is missing label columns: {missing_labels}."
            )

    if feat_cols is not None:
        missing_feats = [col for col in feat_cols if col not in df.columns]
        if missing_feats:
            raise ValueError(
                f"[{stage}] flows data is missing {len(missing_feats)} feature "
                f"column(s), e.g. {missing_feats[:5]}."
            )


def validate_meta(meta: dict[str, Any], stage: str) -> list[str]:
    """Validate metadata dict and return feature column list."""
    for key in ("schema_version", "feature_cols", "clip_upper", "id_columns"):
        if key not in meta:
            raise ValueError(f"[{stage}] meta.json is missing required key: {key!r}.")

    feat_cols = meta["feature_cols"]
    if not feat_cols:
        raise ValueError(f"[{stage}] meta.json feature_cols is empty.")

    expected_dim = meta.get("node_feature_dim")
    if expected_dim is not None:
        computed = node_feature_dim(len(feat_cols))
        if expected_dim != computed:
            raise ValueError(
                f"[{stage}] meta.json node_feature_dim={expected_dim} does not match "
                f"len(feature_cols)+2={computed}."
            )

    return feat_cols


def validate_meta_file(meta_path: Path, stage: str) -> list[str]:
    if not meta_path.exists():
        raise FileNotFoundError(
            f"[{stage}] metadata not found at {meta_path}. Run preprocess first."
        )
    meta = json.loads(meta_path.read_text())
    return validate_meta(meta, stage)


def validate_snapshots(
    snapshots: list[Data],
    feat_cols: list[str],
    stage: str,
) -> None:
    """Verify graph snapshots match expected node feature dimensions."""
    if not snapshots:
        raise ValueError(f"[{stage}] no graph snapshots available.")

    expected_dim = node_feature_dim(len(feat_cols))
    for idx, snapshot in enumerate(snapshots[:3]):
        if snapshot.x.shape[1] != expected_dim:
            raise ValueError(
                f"[{stage}] snapshot {idx} node feature dim "
                f"{snapshot.x.shape[1]} != expected {expected_dim}. "
                "Rebuild snapshots after re-preprocessing."
            )


def validate_checkpoint_matches_features(
    ckpt: dict[str, Any],
    feat_cols: list[str],
    stage: str,
) -> None:
    expected = node_feature_dim(len(feat_cols))
    in_channels = ckpt.get("in_channels")
    if in_channels != expected:
        raise ValueError(
            f"[{stage}] checkpoint in_channels={in_channels} does not match "
            f"current schema node_feature_dim={expected}. Retrain the model."
        )


def build_meta_payload(
    feat_cols: list[str],
    clip_upper: list[float],
) -> dict[str, Any]:
    """Build the canonical meta.json content."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id_columns": list(ID_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
        "feature_cols": feat_cols,
        "clip_upper": clip_upper,
        "node_feature_dim": node_feature_dim(len(feat_cols)),
    }

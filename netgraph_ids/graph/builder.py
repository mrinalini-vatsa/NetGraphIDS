"""
Convert a flow DataFrame into PyTorch Geometric graph snapshots.

Design:
  - Nodes: unique IP addresses observed in the time window
  - Edges: directed flow src_ip → dst_ip
  - Node features: aggregated stats of flows touching each IP
    (mean flow features, in-degree, out-degree)
  - Edge features: per-flow numeric features
  - Node label: 1 if ANY flow involving this node is an attack, else 0
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from netgraph_ids.paths import (
    FLOWS_FILENAME,
    META_FILENAME,
    PROCESSED_DIR,
    SNAPSHOTS_FILENAME,
    ensure_project_dirs,
)
from netgraph_ids.schema import (
    validate_flow_dataframe,
    validate_meta_file,
    validate_snapshots,
)


class IPIndex:
    """Bidirectional map between IP string and integer node index."""

    def __init__(self):
        self._ip2id: dict[str, int] = {}
        self._id2ip: list[str] = []

    def get_or_add(self, ip: str) -> int:
        if ip not in self._ip2id:
            idx = len(self._id2ip)
            self._ip2id[ip] = idx
            self._id2ip.append(ip)
        return self._ip2id[ip]

    def __len__(self):
        return len(self._id2ip)

    def ip(self, idx: int) -> str:
        return self._id2ip[idx]


def _aggregate_node_features(
    df_window: pd.DataFrame,
    ip_index: IPIndex,
    feat_cols: list[str],
) -> torch.Tensor:
    """
    For each node (IP), aggregate incoming + outgoing flow features.
    Returns shape [num_nodes, num_feats + 2]:
       mean of feat_cols, in_degree, out_degree
    """
    num_nodes = len(ip_index)
    num_feats = len(feat_cols)
    feat_arr = np.zeros((num_nodes, num_feats + 2), dtype=np.float32)
    counts = np.zeros(num_nodes, dtype=np.float32)

    for _, row in df_window.iterrows():
        src = ip_index.get_or_add(str(row["src_ip"]))
        dst = ip_index.get_or_add(str(row["dst_ip"]))
        flow_feat = row[feat_cols].values.astype(np.float32)

        feat_arr[src, :num_feats] += flow_feat
        feat_arr[dst, :num_feats] += flow_feat
        counts[src] += 1
        counts[dst] += 1

        feat_arr[src, num_feats + 1] += 1.0
        feat_arr[dst, num_feats] += 1.0

    nonzero = counts > 0
    feat_arr[nonzero, :num_feats] /= counts[nonzero, np.newaxis]

    return torch.tensor(feat_arr, dtype=torch.float)


def build_snapshot(
    df_window: pd.DataFrame,
    feat_cols: list[str],
    ip_index: Optional[IPIndex] = None,
) -> tuple[Data, IPIndex]:
    """Build one PyG Data object from a DataFrame window of flows."""
    if ip_index is None:
        ip_index = IPIndex()

    num_feats = len(feat_cols)
    if df_window.empty:
        empty = Data(
            x=torch.zeros((0, num_feats + 2), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, num_feats), dtype=torch.float),
            y=torch.zeros(0, dtype=torch.long),
        )
        return empty, ip_index

    validate_flow_dataframe(
        df_window,
        stage="graph.build_snapshot",
        require_labels="attack" in df_window.columns,
        feat_cols=feat_cols,
    )

    for ip in pd.concat([df_window["src_ip"], df_window["dst_ip"]]).unique():
        ip_index.get_or_add(str(ip))

    num_nodes = len(ip_index)

    src_ids = df_window["src_ip"].apply(lambda value: ip_index._ip2id[str(value)]).values
    dst_ids = df_window["dst_ip"].apply(lambda value: ip_index._ip2id[str(value)]).values
    edge_index = torch.tensor(np.stack([src_ids, dst_ids], axis=0), dtype=torch.long)

    edge_feat = torch.tensor(
        df_window[feat_cols].values.astype(np.float32), dtype=torch.float
    )

    x = _aggregate_node_features(df_window, ip_index, feat_cols)

    y = torch.zeros(num_nodes, dtype=torch.long)
    if "attack" in df_window.columns:
        for _, row in df_window[df_window["attack"]].iterrows():
            src = ip_index._ip2id.get(str(row["src_ip"]), -1)
            dst = ip_index._ip2id.get(str(row["dst_ip"]), -1)
            if src >= 0:
                y[src] = 1
            if dst >= 0:
                y[dst] = 1

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_feat, y=y)
    return data, ip_index


def generate_snapshots(
    df: pd.DataFrame,
    feat_cols: list[str],
    window_size: int = 500,
    stride: int = 250,
    fresh_index_per_window: bool = True,
) -> Iterator[Data]:
    """Yield graph snapshots by sliding a window over the flow DataFrame."""
    total = len(df)
    start = 0
    while start < total:
        end = min(start + window_size, total)
        window = df.iloc[start:end]
        index = IPIndex() if fresh_index_per_window else None
        snapshot, _ = build_snapshot(window, feat_cols, index)
        if snapshot.num_nodes > 0:
            yield snapshot
        start += stride


def build_and_save(
    processed_dir: Path = PROCESSED_DIR,
    window_size: int = 500,
    stride: int = 250,
) -> Path:
    """Load processed flows, build snapshots, and save as a .pt list."""
    ensure_project_dirs()
    flows_path = processed_dir / FLOWS_FILENAME
    meta_path = processed_dir / META_FILENAME
    out_path = processed_dir / SNAPSHOTS_FILENAME

    if not flows_path.exists():
        raise FileNotFoundError(
            f"Processed flows not found at {flows_path}. Run preprocess first."
        )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}. Run preprocess first."
        )

    df = pd.read_parquet(flows_path)
    feat_cols = validate_meta_file(meta_path, stage="graph.build_and_save")
    validate_flow_dataframe(
        df, stage="graph.build_and_save", feat_cols=feat_cols
    )

    print(f"[graph] Building snapshots  window={window_size}  stride={stride}")
    snapshots = list(generate_snapshots(df, feat_cols, window_size, stride))
    validate_snapshots(snapshots, feat_cols, stage="graph.build_and_save")
    print(f"[graph] Generated {len(snapshots)} snapshots")

    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(snapshots, out_path)
    print(f"[graph] Saved → {out_path}")
    return out_path

"""Data loading and visualization helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from netgraph_ids.detect.engine import DetectionEngine
from netgraph_ids.paths import (
    BEST_MODEL_FILENAME,
    CHECKPOINTS_DIR,
    EVAL_DIR,
    FLOWS_FILENAME,
    PROCESSED_DIR,
)
from netgraph_ids.schema import validate_flow_dataframe


def model_exists() -> bool:
    return (CHECKPOINTS_DIR / BEST_MODEL_FILENAME).exists()


def flows_available() -> bool:
    return (PROCESSED_DIR / FLOWS_FILENAME).exists()


def load_flows(sample_size: int = 5000, seed: int = 42) -> pd.DataFrame:
    path = PROCESSED_DIR / FLOWS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Processed flows not found at {path}. Run preprocess first."
        )
    df = pd.read_parquet(path)
    validate_flow_dataframe(df, stage="dashboard")
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
    return df


def load_eval_metrics() -> Optional[dict[str, Any]]:
    path = EVAL_DIR / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compute_overview_stats(flows: pd.DataFrame) -> dict[str, Any]:
    hosts = pd.concat([flows["src_ip"], flows["dst_ip"]]).nunique()
    attacks = int(flows["attack"].sum()) if "attack" in flows.columns else 0
    attack_dist = (
        flows["attack_type"].value_counts().reset_index()
        if "attack_type" in flows.columns
        else pd.DataFrame(columns=["attack_type", "count"])
    )
    if not attack_dist.empty:
        attack_dist.columns = ["attack_type", "count"]
    return {
        "n_hosts": int(hosts),
        "n_connections": int(len(flows)),
        "n_attacks": attacks,
        "attack_distribution": attack_dist,
    }


def build_network_graph(
    flows: pd.DataFrame,
    predictions: Optional[pd.DataFrame] = None,
    max_edges: int = 300,
) -> tuple[nx.DiGraph, dict[str, str]]:
    """Build a directed graph from flows with optional per-IP predictions."""
    subset = flows.head(max_edges)
    graph = nx.DiGraph()
    node_colors: dict[str, str] = {}

    flagged: set[str] = set()
    if predictions is not None and not predictions.empty:
        flagged = set(
            predictions[predictions["predicted_attack"]]["ip"].astype(str).tolist()
        )

    for _, row in subset.iterrows():
        src = str(row["src_ip"])
        dst = str(row["dst_ip"])
        graph.add_edge(src, dst)
        if src not in node_colors:
            node_colors[src] = "red" if src in flagged else (
                "orange" if row.get("attack", False) else "green"
            )
        if dst not in node_colors:
            node_colors[dst] = "red" if dst in flagged else (
                "orange" if row.get("attack", False) else "green"
            )

    if predictions is not None and not predictions.empty:
        for ip in flagged:
            node_colors[ip] = "red"

    return graph, node_colors


def graph_to_plotly(
    graph: nx.DiGraph,
    node_colors: dict[str, str],
) -> go.Figure:
    """Render a NetworkX graph with Plotly (CPU-friendly spring layout)."""
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No graph data available")
        return fig

    pos = nx.spring_layout(graph, seed=42, k=0.5)
    degrees = dict(graph.degree())

    edge_x, edge_y = [], []
    for src, dst in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.6, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, texts, colors, sizes = [], [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = degrees.get(node, 0)
        colors.append(node_colors.get(node, "green"))
        sizes.append(12 + degree * 3)
        texts.append(f"IP: {node}<br>Degree: {degree}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=texts,
        marker=dict(
            color=colors,
            size=sizes,
            line=dict(width=1, color="#333"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Network Graph (hosts = nodes, flows = edges)",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )
    return fig


def attack_distribution_chart(attack_dist: pd.DataFrame) -> go.Figure:
    if attack_dist.empty:
        return px.bar(title="No attack distribution data")
    fig = px.bar(
        attack_dist,
        x="attack_type",
        y="count",
        color="attack_type",
        title="Attack Type Distribution",
        labels={"attack_type": "Attack Type", "count": "Flow Count"},
    )
    fig.update_layout(showlegend=False, height=400)
    return fig


def confusion_matrix_figure(matrix: list[list[int]]) -> go.Figure:
    arr = np.array(matrix)
    fig = px.imshow(
        arr,
        text_auto=True,
        color_continuous_scale="Blues",
        x=["Predicted Benign", "Predicted Attack"],
        y=["Actual Benign", "Actual Attack"],
        title="GNN Confusion Matrix",
    )
    fig.update_layout(height=400)
    return fig


def run_detection(flows: pd.DataFrame, threshold: float = 0.5) -> dict:
    engine = DetectionEngine.from_checkpoint(threshold=threshold)
    return engine.analyze(flows)


def detection_results_table(node_alerts: pd.DataFrame) -> pd.DataFrame:
    if node_alerts.empty:
        return node_alerts
    table = node_alerts.copy()
    table["confidence_score"] = table["attack_probability"].apply(
        lambda p: max(p, 1 - p)
    )
    table = table.rename(
        columns={
            "ip": "Host IP",
            "attack_probability": "Attack Probability",
            "predicted_attack": "Predicted Attack",
            "confidence_score": "Confidence Score",
        }
    )
    return table.sort_values("Attack Probability", ascending=False)

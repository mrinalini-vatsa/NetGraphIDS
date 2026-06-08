"""
Evaluation suite for NetGraph-IDS.

Compares:
  1. NetGraph GNN (node-level predictions on graph snapshots)
  2. Flat-feature Random Forest (per-flow baseline)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from netgraph_ids.graph.builder import generate_snapshots
from netgraph_ids.model.train import load_model
from netgraph_ids.paths import (
    BEST_MODEL_FILENAME,
    CHECKPOINTS_DIR,
    EVAL_DIR,
    FLOWS_FILENAME,
    META_FILENAME,
    PROCESSED_DIR,
    ensure_project_dirs,
)
from netgraph_ids.schema import validate_flow_dataframe, validate_meta_file


def eval_gnn(
    snapshots: list[Data],
    ckpt_path: Path,
    feat_cols: list[str],
    threshold: float = 0.5,
) -> dict:
    """Run GNN on all snapshots and collect node-level metrics."""
    model, _ = load_model(ckpt_path, feat_cols=feat_cols)
    model.eval()

    all_preds, all_labels, all_proba = [], [], []

    with torch.no_grad():
        for snapshot in snapshots:
            if snapshot.num_nodes == 0:
                continue
            proba = model.predict_proba(snapshot.x, snapshot.edge_index)
            preds = (proba[:, 1] >= threshold).long()
            all_preds.append(preds.numpy())
            all_labels.append(snapshot.y.numpy())
            all_proba.append(proba[:, 1].numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_proba)

    return _compute_metrics("GNN (NetGraph)", y_true, y_pred, y_prob)


def eval_random_forest(
    df: pd.DataFrame,
    feat_cols: list[str],
    seed: int = 42,
) -> tuple[dict, RandomForestClassifier]:
    """Train and evaluate Random Forest on per-flow features."""
    features = df[feat_cols].values.astype(np.float32)
    labels = df["attack"].astype(int).values

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    print("[eval] Training Random Forest baseline …")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(x_train, y_train)
    print(f"[eval] RF trained in {time.time() - t0:.1f}s")

    y_pred = rf.predict(x_test)
    y_prob = rf.predict_proba(x_test)[:, 1]

    return _compute_metrics("Random Forest (Flat)", y_test, y_pred, y_prob), rf


def _compute_metrics(name: str, y_true, y_pred, y_prob) -> dict:
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    ap = average_precision_score(y_true, y_prob)

    return {
        "name": name,
        "f1_attack": report.get("1", {}).get("f1-score", 0.0),
        "precision_attack": report.get("1", {}).get("precision", 0.0),
        "recall_attack": report.get("1", {}).get("recall", 0.0),
        "f1_macro": report.get("macro avg", {}).get("f1-score", 0.0),
        "accuracy": report.get("accuracy", 0.0),
        "roc_auc": float(auc),
        "avg_precision": float(ap),
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def plot_comparison(gnn_metrics: dict, rf_metrics: dict, out_dir: Path):
    """Side-by-side bar chart, confusion matrix, and PR curves."""
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = ["f1_attack", "precision_attack", "recall_attack", "roc_auc"]
    labels = ["F1 (Attack)", "Precision", "Recall", "ROC-AUC"]
    x = np.arange(len(metric_keys))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "NetGraph-IDS: GNN vs Flat-Feature Baseline",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    gnn_vals = [gnn_metrics[key] for key in metric_keys]
    rf_vals = [rf_metrics[key] for key in metric_keys]
    bars1 = ax.bar(
        x - width / 2,
        gnn_vals,
        width,
        label="GNN (NetGraph)",
        color="#2196F3",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2,
        rf_vals,
        width,
        label="Random Forest",
        color="#FF5722",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Detection Metrics Comparison")
    ax.legend()
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax = axes[1]
    cm = np.array(gnn_metrics["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
    )
    ax.set_title("GNN Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax = axes[2]
    for metrics, color in [(gnn_metrics, "#2196F3"), (rf_metrics, "#FF5722")]:
        prec, rec, _ = precision_recall_curve(metrics["y_true"], metrics["y_prob"])
        ap = metrics["avg_precision"]
        ax.plot(rec, prec, color=color, label=f"{metrics['name']}  (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = out_dir / "comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[eval] Plot saved → {out_path}")


def run_evaluation(
    processed_dir: Optional[Path] = None,
    ckpt_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    window_size: int = 500,
    stride: int = 250,
) -> dict:
    """End-to-end evaluation: load data, run GNN + RF, compare, plot."""
    ensure_project_dirs()
    processed_dir = processed_dir or PROCESSED_DIR
    ckpt_path = ckpt_path or (CHECKPOINTS_DIR / BEST_MODEL_FILENAME)
    out_dir = out_dir or EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    flows_path = processed_dir / FLOWS_FILENAME
    meta_path = processed_dir / META_FILENAME

    if not flows_path.exists():
        raise FileNotFoundError(
            f"[eval] flows not found at {flows_path}. Run preprocess first."
        )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"[eval] checkpoint not found at {ckpt_path}. Run train first."
        )

    df = pd.read_parquet(flows_path)
    feat_cols = validate_meta_file(meta_path, stage="eval")
    validate_flow_dataframe(df, stage="eval", feat_cols=feat_cols)

    print("[eval] Building test snapshots …")
    total = len(df)
    test_df = df.iloc[int(total * 0.8) :]

    snapshots = list(generate_snapshots(test_df, feat_cols, window_size, stride))
    snapshots = [snapshot for snapshot in snapshots if snapshot.num_nodes > 0]
    print(f"[eval] {len(snapshots)} test snapshots")

    print("[eval] Evaluating GNN …")
    gnn_metrics = eval_gnn(snapshots, ckpt_path, feat_cols)

    print("[eval] Evaluating Random Forest …")
    rf_metrics, _ = eval_random_forest(test_df, feat_cols)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metrics in [gnn_metrics, rf_metrics]:
        print(f"\n{metrics['name']}")
        print(f"  F1 (Attack)   : {metrics['f1_attack']:.4f}")
        print(f"  Precision     : {metrics['precision_attack']:.4f}")
        print(f"  Recall        : {metrics['recall_attack']:.4f}")
        print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")
        print(f"  Avg Precision : {metrics['avg_precision']:.4f}")
    print("=" * 60)

    gnn_plot = {
        key: value for key, value in gnn_metrics.items() if key not in ("y_true", "y_prob")
    }
    rf_plot = {
        key: value for key, value in rf_metrics.items() if key not in ("y_true", "y_prob")
    }

    plot_comparison(gnn_metrics, rf_metrics, out_dir)

    results = {
        "gnn": gnn_plot,
        "random_forest": rf_plot,
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"[eval] Results saved → {out_dir}/results.json")

    return results

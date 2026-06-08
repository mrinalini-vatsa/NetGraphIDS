"""
Training pipeline for NetGraph-IDS.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from netgraph_ids.model.gnn import NetGraphGNN
from netgraph_ids.paths import (
    BEST_MODEL_FILENAME,
    CHECKPOINTS_DIR,
    META_FILENAME,
    PROCESSED_DIR,
    SNAPSHOTS_FILENAME,
    ensure_project_dirs,
)
from netgraph_ids.schema import (
    node_feature_dim,
    validate_checkpoint_matches_features,
    validate_meta_file,
    validate_snapshots,
)


class SnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, snapshots: list[Data]):
        self.snapshots = snapshots

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        return self.snapshots[idx]


def compute_class_weights(snapshots: list[Data]) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted cross-entropy."""
    counts = torch.zeros(2)
    for snapshot in snapshots:
        counts[0] += (snapshot.y == 0).sum()
        counts[1] += (snapshot.y == 1).sum()
    total = counts.sum()
    return total / (2.0 * counts + 1e-8)


def _run_epoch(
    model: NetGraphGNN,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    train: bool,
) -> dict:
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    node_count = sum(len(item.y) for item in loader.dataset)

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "n_nodes": int(node_count),
    }


def train(
    snapshots_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    hidden_dim: int = 128,
    dropout: float = 0.4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 30,
    batch_size: int = 8,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Path:
    """Full training run. Returns path to best checkpoint."""
    ensure_project_dirs()
    torch.manual_seed(seed)
    np.random.seed(seed)

    snapshots_path = snapshots_path or (PROCESSED_DIR / SNAPSHOTS_FILENAME)
    out_dir = out_dir or CHECKPOINTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not snapshots_path.exists():
        raise FileNotFoundError(
            f"[train] snapshots not found at {snapshots_path}. Run build-graph first."
        )

    feat_cols = validate_meta_file(
        PROCESSED_DIR / META_FILENAME, stage="train"
    )

    print(f"[train] Loading snapshots from {snapshots_path}")
    snapshots: list[Data] = torch.load(snapshots_path, map_location="cpu")
    print(f"[train] {len(snapshots)} snapshots loaded")

    snapshots = [snapshot for snapshot in snapshots if snapshot.num_nodes > 0 and snapshot.num_edges > 0]
    validate_snapshots(snapshots, feat_cols, stage="train")
    if not snapshots:
        raise RuntimeError("No non-empty snapshots available for training.")

    print(f"[train] {len(snapshots)} non-empty snapshots")

    np.random.shuffle(snapshots)
    n_val = max(1, int(len(snapshots) * val_frac))
    val_snaps = snapshots[:n_val]
    train_snaps = snapshots[n_val:]

    print(f"[train] Train={len(train_snaps)}  Val={len(val_snaps)}")

    device = torch.device("cpu")

    train_loader = DataLoader(
        SnapshotDataset(train_snaps), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SnapshotDataset(val_snaps), batch_size=batch_size, shuffle=False
    )

    in_channels = snapshots[0].x.shape[1]
    expected_dim = node_feature_dim(len(feat_cols))
    if in_channels != expected_dim:
        raise ValueError(
            f"[train] snapshot feature dim {in_channels} != "
            f"expected {expected_dim} from meta.json. Rebuild snapshots."
        )
    model = NetGraphGNN(
        in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout
    ).to(device)

    total_params = sum(param.numel() for param in model.parameters())
    print(f"[train] Model: {total_params:,} parameters  |  in_features={in_channels}")

    class_weights = compute_class_weights(train_snaps).to(device)
    print(
        f"[train] Class weights: benign={class_weights[0]:.3f}  "
        f"attack={class_weights[1]:.3f}"
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_f1 = 0.0
    best_ckpt = out_dir / BEST_MODEL_FILENAME
    history = []

    print(
        f"\n{'Epoch':>5}  {'Train Loss':>11}  {'Train F1':>9}  {'Val F1':>7}  "
        f"{'Val Prec':>9}  {'Val Rec':>8}  {'Time':>6}"
    )
    print("-" * 68)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = _run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        val_metrics = _run_epoch(
            model, val_loader, None, criterion, device, train=False
        )

        scheduler.step(val_metrics["f1"])
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_f1": train_metrics["f1"],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        marker = " ← best" if val_metrics["f1"] > best_val_f1 else ""
        print(
            f"{epoch:>5}  {train_metrics['loss']:>11.4f}  {train_metrics['f1']:>9.4f}"
            f"  {val_metrics['f1']:>7.4f}  {val_metrics['precision']:>9.4f}"
            f"  {val_metrics['recall']:>8.4f}  {elapsed:>5.1f}s{marker}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "val_f1": best_val_f1,
                    "history": history,
                },
                best_ckpt,
            )

    print(f"\n[train] Best val F1 = {best_val_f1:.4f}  →  {best_ckpt}")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return best_ckpt


def load_model(
    ckpt_path: Path,
    feat_cols: Optional[list[str]] = None,
) -> tuple[NetGraphGNN, dict]:
    """Load a saved checkpoint. Returns (model, meta_dict)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if feat_cols is not None:
        validate_checkpoint_matches_features(ckpt, feat_cols, stage="load_model")
    model = NetGraphGNN(
        in_channels=ckpt["in_channels"],
        hidden_dim=ckpt["hidden_dim"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

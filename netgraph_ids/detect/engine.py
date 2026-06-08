"""
Inference engine for NetGraph-IDS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from netgraph_ids.data.preprocess import (
    load_clip_upper,
    load_feature_cols,
    load_scaler,
    normalize_columns,
    transform_features,
)
from netgraph_ids.graph.builder import IPIndex, build_snapshot
from netgraph_ids.model.train import load_model
from netgraph_ids.paths import (
    BEST_MODEL_FILENAME,
    CHECKPOINTS_DIR,
    META_FILENAME,
    PROCESSED_DIR,
    ensure_project_dirs,
)
from netgraph_ids.schema import validate_flow_dataframe, validate_meta_file


class DetectionEngine:
    """
    Stateful inference engine.

    Usage:
        engine = DetectionEngine.from_checkpoint()
        alerts = engine.analyze(flow_df)
    """

    def __init__(
        self,
        model,
        feat_cols: list[str],
        scaler,
        clip_upper,
        threshold: float = 0.5,
    ):
        self.model = model
        self.model.eval()
        self.feat_cols = feat_cols
        self.scaler = scaler
        self.clip_upper = clip_upper
        self.threshold = threshold

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        threshold: float = 0.5,
    ) -> "DetectionEngine":
        ensure_project_dirs()
        ckpt_path = ckpt_path or (CHECKPOINTS_DIR / BEST_MODEL_FILENAME)
        processed_dir = processed_dir or PROCESSED_DIR

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")

        feat_cols = validate_meta_file(
            processed_dir / META_FILENAME, stage="inference"
        )
        model, _ = load_model(ckpt_path, feat_cols=feat_cols)
        scaler = load_scaler(processed_dir)
        clip_upper = load_clip_upper(processed_dir)
        return cls(model, feat_cols, scaler, clip_upper, threshold)

    def _prepare_flows(self, flows: pd.DataFrame) -> pd.DataFrame:
        """Normalise columns and apply the saved preprocessing scaler."""
        prepared = normalize_columns(flows.copy())
        prepared = transform_features(
            prepared, self.feat_cols, self.scaler, self.clip_upper
        )

        if "attack" not in prepared.columns:
            prepared["attack"] = False

        validate_flow_dataframe(
            prepared,
            stage="inference",
            require_labels=False,
            feat_cols=self.feat_cols,
        )
        return prepared

    def analyze(
        self,
        flows: pd.DataFrame,
    ) -> dict:
        """
        Run GNN inference on a batch of flows.

        Returns dict with:
          'node_alerts'  : DataFrame — one row per IP, with attack probability
          'flow_alerts'  : DataFrame — flows where either endpoint is flagged
          'summary'      : dict with counts
        """
        if flows.empty:
            return {
                "node_alerts": pd.DataFrame(),
                "flow_alerts": pd.DataFrame(),
                "summary": {"n_nodes": 0, "n_alerts": 0},
            }

        prepared = self._prepare_flows(flows)

        ip_index = IPIndex()
        snapshot, ip_index = build_snapshot(prepared, self.feat_cols, ip_index)

        if snapshot.num_nodes == 0:
            return {
                "node_alerts": pd.DataFrame(),
                "flow_alerts": pd.DataFrame(),
                "summary": {"n_nodes": 0, "n_alerts": 0},
            }

        with torch.no_grad():
            proba = self.model.predict_proba(snapshot.x, snapshot.edge_index)

        attack_proba = proba[:, 1].numpy()
        predicted = (attack_proba >= self.threshold).astype(int)

        node_rows = []
        for idx in range(len(ip_index)):
            node_rows.append(
                {
                    "ip": ip_index.ip(idx),
                    "attack_probability": float(attack_proba[idx]),
                    "predicted_attack": bool(predicted[idx]),
                }
            )
        node_df = pd.DataFrame(node_rows).sort_values(
            "attack_probability", ascending=False
        )

        flagged_ips = set(node_df[node_df["predicted_attack"]]["ip"].tolist())
        flow_alert_mask = (
            prepared["src_ip"].astype(str).isin(flagged_ips)
            | prepared["dst_ip"].astype(str).isin(flagged_ips)
        )
        flow_alerts = prepared[flow_alert_mask].copy()

        summary = {
            "n_nodes": len(node_df),
            "n_alerts": int(predicted.sum()),
            "n_flow_alerts": int(flow_alert_mask.sum()),
            "alert_rate": float(predicted.mean()),
        }

        return {
            "node_alerts": node_df,
            "flow_alerts": flow_alerts,
            "summary": summary,
        }

    def analyze_file(self, path: Path) -> dict:
        """Load a CSV/parquet flow file and run analysis."""
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        return self.analyze(df)

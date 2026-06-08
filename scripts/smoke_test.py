#!/usr/bin/env python3
"""
Generate minimal synthetic CICIDS-style data and run a fast pipeline smoke test.
Useful for verifying imports and end-to-end flow without downloading full CICIDS2017.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from netgraph_ids.data.preprocess import preprocess
from netgraph_ids.eval.compare import run_evaluation
from netgraph_ids.graph.builder import build_and_save
from netgraph_ids.model.train import train
from netgraph_ids.paths import RAW_DIR, ensure_project_dirs


def _make_synthetic_csv(path: Path, n_rows: int, attack_frac: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(n_rows):
        attack = rng.random() < attack_frac
        rows.append(
            {
                "Source IP": f"10.0.{idx % 20}.{(idx // 20) % 255}",
                "Destination IP": f"10.1.{(idx + 7) % 20}.{(idx // 13) % 255}",
                "Source Port": int(rng.integers(1024, 65000)),
                "Destination Port": int(rng.integers(20, 65000)),
                "Protocol": int(rng.integers(6, 18)),
                "Flow Duration": float(rng.exponential(1000)),
                "Total Fwd Packets": float(rng.integers(1, 100)),
                "Total Backward Packets": float(rng.integers(1, 100)),
                "Total Length of Fwd Packets": float(rng.integers(40, 5000)),
                "Total Length of Bwd Packets": float(rng.integers(40, 5000)),
                "Label": "PortScan" if attack else "BENIGN",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    ensure_project_dirs()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = [
        RAW_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        RAW_DIR / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    ]
    for idx, csv_path in enumerate(csv_paths):
        _make_synthetic_csv(csv_path, n_rows=2000, attack_frac=0.2, seed=42 + idx)

    preprocess(sample_benign_frac=1.0, max_attack_rows=10_000, seed=42)
    build_and_save(window_size=200, stride=100)
    train(epochs=2, batch_size=4, val_frac=0.2, seed=42)
    run_evaluation(window_size=200, stride=100)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

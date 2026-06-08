#!/usr/bin/env python3
"""
End-to-end NetGraph-IDS pipeline:
  download -> preprocess -> graph build -> train -> evaluate
"""

from __future__ import annotations

import argparse

from netgraph_ids.data.download import ensure_raw_data
from netgraph_ids.data.preprocess import preprocess
from netgraph_ids.eval.compare import run_evaluation
from netgraph_ids.graph.builder import build_and_save
from netgraph_ids.model.train import train
from netgraph_ids.paths import ensure_project_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full NetGraph-IDS pipeline")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--window-size", type=int, default=500, help="Flows per snapshot")
    parser.add_argument("--stride", type=int, default=250, help="Snapshot stride")
    parser.add_argument(
        "--sample-benign-frac",
        type=float,
        default=0.3,
        help="Fraction of benign flows to keep",
    )
    parser.add_argument(
        "--max-attack-rows",
        type=int,
        default=50_000,
        help="Maximum attack rows to keep",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    print("=" * 60)
    print("STEP 1/5  Download CICIDS2017 raw data")
    print("=" * 60)
    ensure_raw_data()

    print("\n" + "=" * 60)
    print("STEP 2/5  Preprocess flows")
    print("=" * 60)
    preprocess(
        sample_benign_frac=args.sample_benign_frac,
        max_attack_rows=args.max_attack_rows,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("STEP 3/5  Build graph snapshots")
    print("=" * 60)
    build_and_save(window_size=args.window_size, stride=args.stride)

    print("\n" + "=" * 60)
    print("STEP 4/5  Train GraphSAGE model")
    print("=" * 60)
    train(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)

    print("\n" + "=" * 60)
    print("STEP 5/5  Evaluate GNN vs Random Forest")
    print("=" * 60)
    run_evaluation(window_size=args.window_size, stride=args.stride)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

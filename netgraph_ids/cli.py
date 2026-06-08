"""
Command-line interface for NetGraph-IDS.
"""

from __future__ import annotations

from pathlib import Path

import click

from netgraph_ids.data.download import ensure_raw_data
from netgraph_ids.data.preprocess import preprocess
from netgraph_ids.detect.engine import DetectionEngine
from netgraph_ids.eval.compare import run_evaluation
from netgraph_ids.graph.builder import build_and_save
from netgraph_ids.model.train import train
from netgraph_ids.paths import ensure_project_dirs


@click.group()
def main() -> None:
    """NetGraph-IDS: GraphSAGE intrusion detection on CICIDS2017."""
    ensure_project_dirs()


@main.command("download")
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for raw CICIDS2017 CSV files.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-download files even if they already exist.",
)
def download_cmd(raw_dir: Path | None, force: bool) -> None:
    """Download CICIDS2017 raw CSV files."""
    ensure_raw_data(raw_dir, force=force)


@main.command("preprocess")
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing raw CSV files.",
)
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for processed parquet, scaler, and metadata.",
)
@click.option("--sample-benign-frac", default=0.3, show_default=True, type=float)
@click.option("--max-attack-rows", default=50_000, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
def preprocess_cmd(
    raw_dir: Path | None,
    processed_dir: Path | None,
    sample_benign_frac: float,
    max_attack_rows: int,
    seed: int,
) -> None:
    """Clean, subsample, scale, and save flow data."""
    from netgraph_ids.paths import PROCESSED_DIR, RAW_DIR

    preprocess(
        raw_dir=raw_dir or RAW_DIR,
        processed_dir=processed_dir or PROCESSED_DIR,
        sample_benign_frac=sample_benign_frac,
        max_attack_rows=max_attack_rows,
        seed=seed,
    )


@main.command("build-graph")
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing flows.parquet and meta.json.",
)
@click.option("--window-size", default=500, show_default=True, type=int)
@click.option("--stride", default=250, show_default=True, type=int)
def build_graph_cmd(
    processed_dir: Path | None,
    window_size: int,
    stride: int,
) -> None:
    """Build time-windowed graph snapshots from processed flows."""
    from netgraph_ids.paths import PROCESSED_DIR

    build_and_save(
        processed_dir=processed_dir or PROCESSED_DIR,
        window_size=window_size,
        stride=stride,
    )


@main.command("train")
@click.option(
    "--snapshots-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to snapshots.pt.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for model checkpoints.",
)
@click.option("--hidden-dim", default=128, show_default=True, type=int)
@click.option("--dropout", default=0.4, show_default=True, type=float)
@click.option("--lr", default=1e-3, show_default=True, type=float)
@click.option("--epochs", default=30, show_default=True, type=int)
@click.option("--batch-size", default=8, show_default=True, type=int)
@click.option("--val-frac", default=0.2, show_default=True, type=float)
@click.option("--seed", default=42, show_default=True, type=int)
def train_cmd(
    snapshots_path: Path | None,
    out_dir: Path | None,
    hidden_dim: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    val_frac: float,
    seed: int,
) -> None:
    """Train the GraphSAGE node classifier."""
    train(
        snapshots_path=snapshots_path,
        out_dir=out_dir,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        val_frac=val_frac,
        seed=seed,
    )


@main.command("evaluate")
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing processed flows.",
)
@click.option(
    "--ckpt-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to trained model checkpoint.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for evaluation artifacts.",
)
@click.option("--window-size", default=500, show_default=True, type=int)
@click.option("--stride", default=250, show_default=True, type=int)
def evaluate_cmd(
    processed_dir: Path | None,
    ckpt_path: Path | None,
    out_dir: Path | None,
    window_size: int,
    stride: int,
) -> None:
    """Evaluate GNN against a Random Forest baseline."""
    run_evaluation(
        processed_dir=processed_dir,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        window_size=window_size,
        stride=stride,
    )


@main.command("infer")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--ckpt-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to trained model checkpoint.",
)
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing scaler.joblib and meta.json.",
)
@click.option("--threshold", default=0.5, show_default=True, type=float)
def infer_cmd(
    input_path: Path,
    ckpt_path: Path | None,
    processed_dir: Path | None,
    threshold: float,
) -> None:
    """Run inference on a CSV or parquet flow file."""
    engine = DetectionEngine.from_checkpoint(
        ckpt_path=ckpt_path,
        processed_dir=processed_dir,
        threshold=threshold,
    )
    result = engine.analyze_file(input_path)
    summary = result["summary"]
    click.echo(f"Nodes analyzed: {summary['n_nodes']}")
    click.echo(f"Alerts raised:  {summary['n_alerts']}")
    click.echo(f"Flow alerts:    {summary['n_flow_alerts']}")
    if not result["node_alerts"].empty:
        click.echo("\nTop suspicious IPs:")
        click.echo(result["node_alerts"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

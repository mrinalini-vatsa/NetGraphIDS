"""Central path definitions for NetGraph-IDS."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
EVAL_DIR = PROJECT_ROOT / "eval_results"

SCALER_FILENAME = "scaler.joblib"
META_FILENAME = "meta.json"
FLOWS_FILENAME = "flows.parquet"
SNAPSHOTS_FILENAME = "snapshots.pt"
BEST_MODEL_FILENAME = "best_model.pt"


def ensure_project_dirs() -> None:
    """Create standard project directories if they do not exist."""
    for path in (RAW_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, EVAL_DIR):
        path.mkdir(parents=True, exist_ok=True)

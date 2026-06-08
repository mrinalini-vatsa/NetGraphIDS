"""
Clean and normalise CICIDS2017 flow CSV files into a unified DataFrame.

Output schema:
  src_ip, dst_ip, src_port, dst_port, protocol,
  timestamp,
  label,
  attack,
  attack_type,
  + numeric flow features (MinMax-scaled to [0, 1])
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from netgraph_ids.paths import (
    FLOWS_FILENAME,
    META_FILENAME,
    PROCESSED_DIR,
    RAW_DIR,
    SCALER_FILENAME,
    ensure_project_dirs,
)
from netgraph_ids.schema import (
    COLUMN_ALIASES,
    NON_FEATURE_COLUMNS,
    build_meta_payload,
    validate_flow_dataframe,
)

warnings.filterwarnings("ignore")

LABEL_MAP = {
    "BENIGN": "benign",
    "PortScan": "portscan",
    "DoS Hulk": "dos",
    "DoS GoldenEye": "dos",
    "DoS slowloris": "dos",
    "DoS Slowhttptest": "dos",
    "DoS attacks-Hulk": "dos",
    "DoS attacks-GoldenEye": "dos",
    "DoS attacks-Slowloris": "dos",
    "DoS attacks-SlowHTTPTest": "dos",
    "Bot": "botnet",
    "DDoS": "dos",
    "Heartbleed": "other",
    "Web Attack \x96 Brute Force": "other",
    "Web Attack \x96 XSS": "other",
    "Web Attack \x96 Sql Injection": "other",
    "FTP-Patator": "other",
    "SSH-Patator": "other",
    "Infiltration": "other",
}


def _norm_col(column: str) -> str:
    column = column.strip().lower()
    column = re.sub(r"\s+", "_", column)
    column = re.sub(r"[^a-z0-9_]", "", column)
    return column


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to match the preprocessing schema."""
    df = df.copy()
    df.columns = [_norm_col(column) for column in df.columns]
    df.rename(
        columns={key: value for key, value in COLUMN_ALIASES.items() if key in df.columns},
        inplace=True,
    )
    return df


def _load_one(path: Path) -> pd.DataFrame:
    """Load a single CICIDS CSV, normalise column names, parse IPs."""
    print(f"  Loading {path.name} …")
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df = normalize_columns(df)

    if "label" not in df.columns:
        raise ValueError(f"No 'label' column found in {path.name}")

    missing_ips = [col for col in ("src_ip", "dst_ip") if col not in df.columns]
    if missing_ips:
        raise ValueError(
            f"{path.name} is missing {missing_ips} after column normalization. "
            "MachineLearningCSV files lack IP columns. "
            "Re-download with: netgraph-ids download --force"
        )

    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/nan, drop rows with missing IPs."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in ["src_ip", "dst_ip"]:
        if col in df.columns:
            df = df[df[col].notna() & (df[col] != "")]

    df.dropna(axis=1, how="all", inplace=True)
    df.fillna(0, inplace=True)
    return df


def _add_attack_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean attack flag and coarse attack_type."""
    df["attack"] = df["label"].apply(lambda value: value.strip() != "BENIGN")
    df["attack_type"] = df["label"].apply(
        lambda value: LABEL_MAP.get(value.strip(), "other")
    )
    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (exclude ID + label cols)."""
    feats = []
    for column in df.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if df[column].dtype in [np.float64, np.float32, np.int64, np.int32, np.uint8]:
            feats.append(column)
    return feats


def _clip_outliers(arr: np.ndarray, upper: Optional[np.ndarray] = None) -> np.ndarray:
    """Clip extreme outliers to the 99.9th percentile per feature."""
    if upper is None:
        upper = np.percentile(arr, 99.9, axis=0)
    return np.clip(arr, 0, upper)


def save_scaler(scaler: MinMaxScaler, processed_dir: Path) -> Path:
    """Persist the fitted MinMaxScaler for inference."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = processed_dir / SCALER_FILENAME
    joblib.dump(scaler, scaler_path)
    return scaler_path


def load_scaler(processed_dir: Path = PROCESSED_DIR) -> MinMaxScaler:
    """Load the fitted MinMaxScaler from disk."""
    scaler_path = processed_dir / SCALER_FILENAME
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Run preprocessing first."
        )
    return joblib.load(scaler_path)


def load_feature_cols(processed_dir: Path = PROCESSED_DIR) -> list[str]:
    """Load feature column names from metadata."""
    meta_path = processed_dir / META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}. Run preprocessing first."
        )
    return json.loads(meta_path.read_text())["feature_cols"]


def load_clip_upper(processed_dir: Path = PROCESSED_DIR) -> np.ndarray:
    """Load saved per-feature clip upper bounds."""
    meta = json.loads((processed_dir / META_FILENAME).read_text())
    return np.array(meta["clip_upper"], dtype=np.float32)


def transform_features(
    df: pd.DataFrame,
    feat_cols: list[str],
    scaler: MinMaxScaler,
    clip_upper: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Apply the same clipping and scaling used during preprocessing."""
    df = df.copy()
    for column in feat_cols:
        if column not in df.columns:
            df[column] = 0.0

    arr = df[feat_cols].values.astype(np.float32)
    arr = _clip_outliers(arr, clip_upper)
    df[feat_cols] = scaler.transform(arr)
    return df


def _fit_scale(
    df: pd.DataFrame, feat_cols: list[str]
) -> tuple[pd.DataFrame, MinMaxScaler, np.ndarray]:
    """Fit MinMaxScaler on training data and transform feature columns."""
    arr = df[feat_cols].values.astype(np.float32)
    clip_upper = np.percentile(arr, 99.9, axis=0).astype(np.float32)
    arr = _clip_outliers(arr, clip_upper)
    scaler = MinMaxScaler()
    df[feat_cols] = scaler.fit_transform(arr)
    return df, scaler, clip_upper


def preprocess(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    sample_benign_frac: float = 0.3,
    max_attack_rows: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Saves:
      - flows.parquet
      - meta.json (feature column list)
      - scaler.joblib (fitted MinMaxScaler)
    """
    ensure_project_dirs()
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / FLOWS_FILENAME

    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}. Run download first.")

    print(f"[preprocess] Found {len(csvs)} CSV(s) in {raw_dir}")
    frames = []
    for path in csvs:
        try:
            frame = _load_one(path)
            frame = _clean(frame)
            frame = _add_attack_labels(frame)
            frames.append(frame)
        except Exception as exc:
            print(f"  [WARN] Skipping {path.name}: {exc}")

    if not frames:
        raise RuntimeError("No files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"[preprocess] Combined: {len(combined):,} rows")

    benign = combined[~combined["attack"]]
    attacks = combined[combined["attack"]]

    benign_sample = benign.sample(frac=sample_benign_frac, random_state=seed)
    attack_sample = (
        attacks
        if len(attacks) <= max_attack_rows
        else attacks.sample(n=max_attack_rows, random_state=seed)
    )

    df = pd.concat([benign_sample, attack_sample], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"[preprocess] After subsampling: {len(df):,} rows")
    print(f"             Attack dist:\n{df['attack_type'].value_counts()}")

    feat_cols = _get_feature_cols(df)
    df, scaler, clip_upper = _fit_scale(df, feat_cols)
    print(f"[preprocess] Feature columns: {len(feat_cols)}")

    validate_flow_dataframe(df, stage="preprocess", feat_cols=feat_cols)

    df.to_parquet(out_path, index=False)
    print(f"[preprocess] Saved → {out_path}")

    meta = build_meta_payload(feat_cols, clip_upper.tolist())
    (processed_dir / META_FILENAME).write_text(json.dumps(meta, indent=2))
    scaler_path = save_scaler(scaler, processed_dir)
    print(f"[preprocess] Scaler saved → {scaler_path}")

    return df

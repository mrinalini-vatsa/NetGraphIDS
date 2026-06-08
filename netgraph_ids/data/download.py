"""
Download CICIDS2017 CSV files and verify integrity.

Uses GeneratedLabelledFlows parquet files from the Ariasyah/cic-ids-2017
Hugging Face mirror. These files include Source IP and Destination IP,
which are required for graph construction.

The c01dsnap MachineLearningCSV mirror does NOT include IP columns and
must not be used with this pipeline.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from netgraph_ids.paths import RAW_DIR, ensure_project_dirs
from netgraph_ids.schema import RAW_IP_HEADERS, validate_raw_csv_header

HF_DATASET_BASE = (
    "https://huggingface.co/datasets/Ariasyah/cic-ids-2017/resolve/main/traffic_labels"
)


@dataclass(frozen=True)
class DatasetFile:
    filename: str
    parquet_name: str
    attack_classes: tuple[str, ...]
    url: str
    sha256: str
    size_bytes: int


DATASET_FILES: tuple[DatasetFile, ...] = (
    DatasetFile(
        filename="Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        parquet_name="Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
        attack_classes=("Port Scan",),
        url=(
            f"{HF_DATASET_BASE}/"
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet?download=true"
        ),
        sha256="4d78cee297c27f1a9947b9384793e587a46c7a3ea89db199553dabddc9835d4a",
        size_bytes=18_632_427,
    ),
    DatasetFile(
        filename="Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        parquet_name="Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
        attack_classes=("DoS/DDoS",),
        url=(
            f"{HF_DATASET_BASE}/"
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet?download=true"
        ),
        sha256="7c5876d52189fc01af54bad6cf23afe9f7fbc0e3ca6c3595920754f0c3ba8f66",
        size_bytes=23_048_086,
    ),
    DatasetFile(
        filename="Friday-WorkingHours-Morning.pcap_ISCX.csv",
        parquet_name="Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
        attack_classes=("DoS/DDoS", "Botnet"),
        url=(
            f"{HF_DATASET_BASE}/"
            "Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet?download=true"
        ),
        sha256="2c00236b13a69f4b1c222b8f4a89451dc2148cb04a8cd0c45c2a87af51471774",
        size_bytes=21_999_571,
    ),
)

TARGET_FILES = [entry.filename for entry in DATASET_FILES]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def list_download_urls() -> dict[str, str]:
    """Return mapping of output CSV filename → download URL."""
    return {entry.filename: entry.url for entry in DATASET_FILES}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_bytes(url: str, expected_size: int, label: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NetGraph-IDS/1.0 (CICIDS2017 downloader)"},
    )
    buffer = io.BytesIO()
    with urllib.request.urlopen(request, timeout=300) as response:
        total = int(response.headers.get("Content-Length", expected_size))
        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=label,
            total=total,
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
                progress.update(len(chunk))
    return buffer.getvalue()


def _parquet_to_csv(data: bytes, entry: DatasetFile, dest: Path) -> None:
    """Convert verified GeneratedLabelledFlows parquet to CSV in raw/."""
    frame = pd.read_parquet(io.BytesIO(data))
    for header in RAW_IP_HEADERS:
        if header not in frame.columns:
            raise ValueError(
                f"{entry.filename} parquet is missing {header!r}. "
                "Expected GeneratedLabelledFlows schema."
            )
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    validate_raw_csv_header(dest.read_text(encoding="utf-8", errors="replace").splitlines()[0], dest.name)


def verify_file(path: Path, entry: DatasetFile) -> bool:
    """Verify CSV exists with correct header and IP columns."""
    if not path.exists():
        return False
    try:
        header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        validate_raw_csv_header(header, path.name)
        return True
    except ValueError as exc:
        print(f"  [verify] {path.name}: {exc}")
        return False


def ensure_raw_data(data_dir: Optional[Path] = None, force: bool = False) -> Path:
    """
    Ensure raw CSVs exist. Returns the raw data directory.
    Downloads GeneratedLabelledFlows parquet, verifies SHA-256, writes CSV.
    """
    ensure_project_dirs()
    raw = data_dir or RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)

    to_fetch: list[DatasetFile] = []
    for entry in DATASET_FILES:
        dest = raw / entry.filename
        if force or not verify_file(dest, entry):
            to_fetch.append(entry)

    if not to_fetch:
        print(f"[data] All {len(DATASET_FILES)} raw files present and verified in {raw}")
        _print_attack_coverage()
        return raw

    print(f"[data] Fetching {len(to_fetch)} GeneratedLabelledFlows file(s) …")
    print("       Source: https://huggingface.co/datasets/Ariasyah/cic-ids-2017")
    print("       (Includes Source IP / Destination IP required for graph building)")
    print()

    for entry in to_fetch:
        dest = raw / entry.filename
        print(f"  Downloading: {entry.filename}")
        print(f"    URL: {entry.url}")
        print(f"    Attack classes: {', '.join(entry.attack_classes)}")
        try:
            payload = _download_bytes(entry.url, entry.size_bytes, entry.parquet_name)
            digest = _sha256_bytes(payload)
            if digest != entry.sha256:
                raise ValueError(
                    f"SHA-256 mismatch: expected {entry.sha256}, got {digest}"
                )
            print(f"    SHA-256 verified: {entry.sha256[:16]}…")
            if dest.exists():
                dest.unlink()
            _parquet_to_csv(payload, entry, dest)
            if not verify_file(dest, entry):
                dest.unlink(missing_ok=True)
                raise ValueError("CSV verification failed after conversion")
        except Exception as exc:
            print(f"  [ERROR] Could not download {entry.filename}: {exc}")
            print()
            print("  Manual options:")
            print("  1. Hugging Face: https://huggingface.co/datasets/Ariasyah/cic-ids-2017")
            print("     (traffic_labels/ folder — must include Source IP & Destination IP)")
            print("  2. Official UNB GeneratedLabelledFlows.zip")
            print(f"  Place verified CSV files in: {raw}/")
            for file_entry in DATASET_FILES:
                print(f"    - {file_entry.filename}")
            print()
            raise FileNotFoundError(
                f"Failed to download {entry.filename}. See instructions above."
            ) from exc

    print(f"\n[data] All files downloaded and verified in {raw}")
    _print_attack_coverage()
    return raw


def _print_attack_coverage() -> None:
    classes = sorted({cls for entry in DATASET_FILES for cls in entry.attack_classes})
    print(f"[data] Attack class coverage: {', '.join(classes)}")

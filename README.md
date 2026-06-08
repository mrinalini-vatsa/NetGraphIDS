# NetGraph-IDS

GraphSAGE-based intrusion detection on **CICIDS2017** network flow data. This project converts flow records into time-windowed IP graphs and trains a node classifier to flag malicious hosts. A flat Random Forest baseline is included for comparison.

## Architecture

```
Raw CICIDS2017 CSVs
        │
        ▼
   Preprocessing  ──► flows.parquet, meta.json, scaler.joblib
        │
        ▼
   Graph Builder  ──► snapshots.pt (sliding-window PyG graphs)
        │
        ▼
   GraphSAGE GNN  ──► checkpoints/best_model.pt
        │
        ├── Evaluation vs Random Forest
        └── Inference engine (DetectionEngine)
```

**Graph design**

| Component | Description |
|-----------|-------------|
| Nodes | Unique IP addresses in a time window |
| Edges | Directed flows `src_ip → dst_ip` |
| Node features | Mean flow features, in-degree, out-degree |
| Node labels | 1 if the IP participates in any attack flow |
| Model | 3-layer GraphSAGE (`SAGEConv`) node classifier |

Node features intentionally exclude label-derived signals such as `attack_neighbor_ratio` to avoid leakage.

## Requirements

- Python 3.10+
- CPU-only compatible (no GPU required)
- ~2 GB disk space for a subset of CICIDS2017

## Setup

```bash
cd /path/to/Mrinalini_Project

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

PyTorch Geometric wheels may require the PyG wheel index on some platforms:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
pip install -r requirements.txt
pip install -e .
```

## Project Layout

```
Mrinalini_Project/
├── app.py                  # Streamlit dashboard entry point
├── netgraph_ids/
│   ├── cli.py              # CLI entrypoint
│   ├── paths.py            # Central path constants
│   ├── dashboard/          # Streamlit helpers
│   ├── data/
│   │   ├── download.py
│   │   └── preprocess.py
│   ├── graph/
│   │   └── builder.py
│   ├── model/
│   │   ├── gnn.py
│   │   └── train.py
│   ├── detect/
│   │   └── engine.py
│   └── eval/
│       └── compare.py
├── data/
│   ├── raw/                # CICIDS2017 CSV files
│   └── processed/          # flows.parquet, scaler, snapshots
├── checkpoints/            # trained model weights
├── eval_results/           # metrics and plots
├── docs/screenshots/       # dashboard screenshots (optional)
├── run_pipeline.py         # full end-to-end script
├── requirements.txt
└── pyproject.toml
```

Directories under `data/`, `checkpoints/`, and `eval_results/` are created automatically on first run.

## Quick Start (Full Pipeline)

```bash
python run_pipeline.py
```

### Fast local smoke test (no download)

Generates tiny synthetic CSVs and runs preprocess → graph → train (2 epochs) → evaluate:

```bash
python scripts/smoke_test.py
```

Optional flags:

```bash
python run_pipeline.py --epochs 10 --batch-size 4 --window-size 500 --stride 250
```

## Step-by-Step CLI

```bash
# 1. Download CICIDS2017 subset
netgraph-ids download

# 2. Preprocess flows (saves scaler.joblib + meta.json)
netgraph-ids preprocess

# 3. Build graph snapshots
netgraph-ids build-graph

# 4. Train GraphSAGE model
netgraph-ids train --epochs 30 --batch-size 8

# 5. Evaluate GNN vs Random Forest
netgraph-ids evaluate
```

Equivalent module invocation:

```bash
python -m netgraph_ids.cli download
python -m netgraph_ids.cli preprocess
python -m netgraph_ids.cli build-graph
python -m netgraph_ids.cli train
python -m netgraph_ids.cli evaluate
```

## Training

Training reads `data/processed/snapshots.pt`, splits snapshots 80/20 train/val, and saves the best checkpoint by validation F1 to `checkpoints/best_model.pt`.

```bash
netgraph-ids train --epochs 30 --hidden-dim 128 --dropout 0.4 --lr 0.001
```

Training history is written to `checkpoints/history.json`.

## Evaluation

Evaluation rebuilds snapshots from the last 20% of processed flows and compares:

1. **GNN (NetGraph)** — node-level predictions on graph snapshots
2. **Random Forest** — flat per-flow baseline

Outputs:

- `eval_results/results.json`
- `eval_results/comparison.png`

```bash
netgraph-ids evaluate
```

## Inference

Run detection on a CSV or parquet file with the same CICIDS-style columns:

```bash
netgraph-ids infer data/processed/flows.parquet --threshold 0.5
```

Python API:

```python
from netgraph_ids.detect import DetectionEngine

engine = DetectionEngine.from_checkpoint()
result = engine.analyze_file("data/processed/flows.parquet")
print(result["summary"])
print(result["node_alerts"].head())
```

The inference engine loads `scaler.joblib` and applies the same clipping/scaling used during preprocessing.

## Dashboard (Streamlit)

After training completes (`checkpoints/best_model.pt` exists), launch the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard runs entirely on CPU and includes:

| Page | Description |
|------|-------------|
| **Overview** | Host count, connection count, attack flows, distribution chart |
| **Network Graph** | Interactive host/flow graph (green=benign, red=malicious) |
| **Detection Results** | Suspicious hosts with attack probability and confidence |
| **Model Metrics** | Accuracy, precision, recall, F1, confusion matrix |
| **Upload & Detect** | Upload a CSV and run live inference |

### Sample Screenshots

Capture screenshots after launching the dashboard and save them to `docs/screenshots/`:

```
docs/screenshots/
├── overview.png
├── network_graph.png
├── detection_results.png
├── model_metrics.png
└── upload_detect.png
```

Example markdown for your portfolio README:

```markdown
![Overview](docs/screenshots/overview.png)
![Network Graph](docs/screenshots/network_graph.png)
```

## Data Download Sources

Automatic downloads use **GeneratedLabelledFlows** parquet files from the Ariasyah Hugging Face mirror (includes Source IP / Destination IP required for graph building):

| File | Attack Classes | Source |
|------|----------------|--------|
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Port Scan | [Ariasyah/traffic_labels](https://huggingface.co/datasets/Ariasyah/cic-ids-2017) |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | DoS/DDoS | [Ariasyah/traffic_labels](https://huggingface.co/datasets/Ariasyah/cic-ids-2017) |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | DoS/DDoS, Botnet | [Ariasyah/traffic_labels](https://huggingface.co/datasets/Ariasyah/cic-ids-2017) |

> **Note:** The c01dsnap MachineLearningCSV mirror strips IP columns and cannot be used with this graph-based pipeline.

Downloads are verified with SHA-256 checksums before use.

If automatic download fails, place the three CSV files in `data/raw/` manually from [Hugging Face](https://huggingface.co/datasets/c01dsnap/CIC-IDS2017) or the [official UNB page](https://www.unb.ca/cic/datasets/ids-2017.html).

## License

Research and educational use. CICIDS2017 is provided by the Canadian Institute for Cybersecurity (UNB).

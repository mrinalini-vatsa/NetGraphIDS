#!/usr/bin/env python3
"""
NetGraph-IDS Streamlit dashboard.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from netgraph_ids.dashboard.helpers import (
    attack_distribution_chart,
    build_network_graph,
    compute_overview_stats,
    confusion_matrix_figure,
    detection_results_table,
    flows_available,
    graph_to_plotly,
    load_eval_metrics,
    load_flows,
    model_exists,
    run_detection,
)
from netgraph_ids.paths import CHECKPOINTS_DIR, EVAL_DIR, PROCESSED_DIR, ensure_project_dirs

st.set_page_config(
    page_title="NetGraph-IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
)

ensure_project_dirs()


def _require_model() -> bool:
    if model_exists():
        return True
    st.warning(
        "No trained model found. Run the pipeline first:\n\n"
        "```bash\npython run_pipeline.py\n```\n\n"
        f"Expected checkpoint: `{CHECKPOINTS_DIR / 'best_model.pt'}`"
    )
    return False


def _require_flows() -> bool:
    if flows_available():
        return True
    st.warning(
        "No processed flows found. Run preprocessing first:\n\n"
        "```bash\nnetgraph-ids download\nnetgraph-ids preprocess\n```"
    )
    return False


def page_overview() -> None:
    st.header("Overview")
    if not _require_flows():
        return

    sample_size = st.slider("Sample size (flows)", 500, 10000, 3000, step=500)
    flows = load_flows(sample_size=sample_size)
    stats = compute_overview_stats(flows)

    col1, col2, col3 = st.columns(3)
    col1.metric("Hosts", f"{stats['n_hosts']:,}")
    col2.metric("Connections", f"{stats['n_connections']:,}")
    col3.metric("Detected Attack Flows", f"{stats['n_attacks']:,}")

    st.plotly_chart(
        attack_distribution_chart(stats["attack_distribution"]),
        use_container_width=True,
    )


def page_network_graph() -> None:
    st.header("Network Graph Visualization")
    if not _require_flows():
        return

    max_edges = st.slider("Max flows to visualize", 50, 500, 200, step=50)
    flows = load_flows(sample_size=max_edges * 2)
    predictions = None

    if model_exists():
        threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)
        if st.button("Run detection on sample", type="primary"):
            with st.spinner("Running GraphSAGE inference …"):
                result = run_detection(flows.head(max_edges), threshold=threshold)
                predictions = result["node_alerts"]
                st.session_state["graph_predictions"] = predictions
        predictions = st.session_state.get("graph_predictions")

    graph, node_colors = build_network_graph(
        flows.head(max_edges),
        predictions=predictions,
        max_edges=max_edges,
    )
    st.plotly_chart(graph_to_plotly(graph, node_colors), use_container_width=True)
    st.caption(
        "Green = benign host, Orange = ground-truth attack flow endpoint, "
        "Red = model-predicted malicious host. Node size ∝ degree."
    )


def page_detection_results() -> None:
    st.header("Detection Results")
    if not _require_flows() or not _require_model():
        return

    sample_size = st.slider("Flows to analyze", 200, 5000, 1000, step=200)
    threshold = st.slider("Alert threshold", 0.1, 0.9, 0.5, 0.05)

    flows = load_flows(sample_size=sample_size)
    with st.spinner("Running detection …"):
        result = run_detection(flows, threshold=threshold)

    summary = result["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes analyzed", summary["n_nodes"])
    c2.metric("Alerts raised", summary["n_alerts"])
    c3.metric("Alert rate", f"{summary['alert_rate']:.1%}")

    table = detection_results_table(result["node_alerts"])
    suspicious = table[table["Predicted Attack"]] if not table.empty else table
    st.subheader("Suspicious Hosts")
    st.dataframe(suspicious, use_container_width=True, hide_index=True)


def page_model_metrics() -> None:
    st.header("Model Metrics")
    metrics = load_eval_metrics()
    if metrics is None:
        st.warning(
            "No evaluation results found. Run:\n\n"
            "```bash\nnetgraph-ids evaluate\n```\n\n"
            f"Expected file: `{EVAL_DIR / 'results.json'}`"
        )
        return

    gnn = metrics.get("gnn", {})
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{gnn.get('accuracy', 0):.3f}")
    cols[1].metric("Precision", f"{gnn.get('precision_attack', 0):.3f}")
    cols[2].metric("Recall", f"{gnn.get('recall_attack', 0):.3f}")
    cols[3].metric("F1 Score", f"{gnn.get('f1_attack', 0):.3f}")

    if "confusion_matrix" in gnn:
        st.plotly_chart(
            confusion_matrix_figure(gnn["confusion_matrix"]),
            use_container_width=True,
        )

    st.subheader("GNN vs Random Forest")
    rf = metrics.get("random_forest", {})
    comparison = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            "GNN": [
                gnn.get("accuracy", 0),
                gnn.get("precision_attack", 0),
                gnn.get("recall_attack", 0),
                gnn.get("f1_attack", 0),
                gnn.get("roc_auc", 0),
            ],
            "Random Forest": [
                rf.get("accuracy", 0),
                rf.get("precision_attack", 0),
                rf.get("recall_attack", 0),
                rf.get("f1_attack", 0),
                rf.get("roc_auc", 0),
            ],
        }
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    comparison_png = EVAL_DIR / "comparison.png"
    if comparison_png.exists():
        st.image(str(comparison_png), caption="Evaluation comparison plot")


def page_upload() -> None:
    st.header("Upload & Detect")
    if not _require_model():
        return

    uploaded = st.file_uploader("Upload a CICIDS-style CSV", type=["csv"])
    threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)

    if uploaded is None:
        st.info("Upload a flow CSV to run GraphSAGE detection.")
        return

    if st.button("Run detection", type="primary"):
        raw_bytes = uploaded.getvalue()
        flows = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
        st.write(f"Loaded **{len(flows):,}** flows from `{uploaded.name}`")

        with st.spinner("Analyzing …"):
            result = run_detection(flows, threshold=threshold)

        summary = result["summary"]
        st.success(
            f"Analysis complete — {summary['n_alerts']} alerts on "
            f"{summary['n_nodes']} hosts."
        )

        table = detection_results_table(result["node_alerts"])
        st.subheader("Suspicious Hosts")
        suspicious = table[table["Predicted Attack"]] if not table.empty else table
        st.dataframe(suspicious, use_container_width=True, hide_index=True)

        if not result["flow_alerts"].empty:
            st.subheader("Flagged Flows")
            st.dataframe(
                result["flow_alerts"].head(100),
                use_container_width=True,
                hide_index=True,
            )


PAGES = {
    "Overview": page_overview,
    "Network Graph": page_network_graph,
    "Detection Results": page_detection_results,
    "Model Metrics": page_model_metrics,
    "Upload & Detect": page_upload,
}

with st.sidebar:
    st.title("NetGraph-IDS")
    st.caption("GraphSAGE intrusion detection on CICIDS2017")
    selection = st.radio("Navigation", list(PAGES.keys()))
    st.divider()
    st.markdown("**Pipeline status**")
    st.write("✅ Model" if model_exists() else "❌ Model")
    st.write("✅ Flows" if flows_available() else "❌ Flows")
    st.write("✅ Metrics" if load_eval_metrics() else "❌ Metrics")

PAGES[selection]()

"""
IoT Shield — Redesigned Dashboard
Run with: python -m streamlit run step4_dashboard.py
"""

import json
import subprocess
import time
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("data")
ALERTS_FILE = DATA_DIR / "live_alerts.json"
STATS_FILE = DATA_DIR / "live_stats.json"

SEVERITY_MAP = {
    "ddos": ("critical", "🔴"),
    "malware": ("critical", "🔴"),
    "portscan": ("high", "🟡"),
    "benign": ("none", "🟢"),
}

st.set_page_config(
    page_title="IoT Shield — Network Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — dark SOC-style theme
# ============================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
    }
    .critical-alert {
        background: rgba(255, 0, 0, 0.08);
        border-left: 4px solid #ff4b4b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-family: 'Courier New', monospace;
    }
    .high-alert {
        background: rgba(255, 193, 7, 0.08);
        border-left: 4px solid #ffc107;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-family: 'Courier New', monospace;
    }
    .benign-alert {
        background: rgba(0, 200, 0, 0.05);
        border-left: 4px solid #2ecc71;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-family: 'Courier New', monospace;
        opacity: 0.7;
    }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def reset_dashboard():
    DATA_DIR.mkdir(exist_ok=True)
    with open(ALERTS_FILE, "w") as f:
        json.dump([], f)
    with open(STATS_FILE, "w") as f:
        json.dump({}, f)


def get_class_counts(alerts):
    counts = Counter(a.get("label", "benign") for a in alerts)
    for cls in ["benign", "ddos", "malware", "portscan"]:
        counts.setdefault(cls, 0)
    return counts


# ============================================================
# SIDEBAR — controls
# ============================================================
st.sidebar.title("🛡️ IoT Shield")
st.sidebar.caption("Real-time network threat detection")
st.sidebar.divider()

if "sim_process" not in st.session_state:
    st.session_state.sim_process = None

col_a, col_b = st.sidebar.columns(2)

if col_a.button("▶ Start Demo", use_container_width=True):
    if st.session_state.sim_process is None:
        st.session_state.sim_process = subprocess.Popen(
            ["python", "step5_simulate.py", "--reset"]
        )
        st.sidebar.success("Simulation started")

if col_b.button("⏸ Stop", use_container_width=True):
    if st.session_state.sim_process is not None:
        st.session_state.sim_process.terminate()
        st.session_state.sim_process = None
        st.sidebar.info("Simulation stopped")

if st.sidebar.button("🔄 Reset Dashboard", use_container_width=True):
    reset_dashboard()
    st.sidebar.success("Dashboard reset")

st.sidebar.divider()
auto_refresh = st.sidebar.toggle("Auto-refresh (5s)", value=True)
severity_filter = st.sidebar.multiselect(
    "Filter alerts by class",
    ["benign", "ddos", "malware", "portscan"],
    default=["ddos", "malware", "portscan"],
)

st.sidebar.divider()
st.sidebar.caption("Model: RF + XGBoost soft-voting ensemble")
st.sidebar.caption("Accuracy: 100.00% (test set)")

# ============================================================
# HEADER
# ============================================================
st.title("🛡️ IoT Shield — Network Threat Detection")
st.caption("Live monitoring dashboard · IoT-23 dataset · Ensemble ML detection")

alerts = load_json(ALERTS_FILE, [])
stats = load_json(STATS_FILE, {})
counts = get_class_counts(alerts)
total_packets = stats.get("total_packets", sum(counts.values()))

with st.status("Monitoring network traffic...", expanded=False) as status:
    st.write(f"Packets processed: {total_packets}")
    status.update(label=f"Monitoring active — {total_packets} packets analyzed", state="running")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Live Monitor", "📊 Analytics", "🚨 Alerts", "🤖 AI Assistant"]
)

# ------------------------------------------------------------
# TAB 1 — Live Monitor
# ------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Benign", counts["benign"])
    c2.metric("🟡 Port Scan", counts["portscan"])
    c3.metric("🔴 DDoS", counts["ddos"])
    c4.metric("🔴 Malware", counts["malware"])

    st.divider()

    # Rolling traffic chart
    if alerts:
        df = pd.DataFrame(alerts)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            fig = px.area(
                df.tail(200),
                x="timestamp",
                color="label",
                title="Traffic Classification Over Time",
                color_discrete_map={
                    "benign": "#2ecc71",
                    "ddos": "#ff4b4b",
                    "malware": "#e74c3c",
                    "portscan": "#ffc107",
                },
            )
            fig.update_layout(
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font_color="white",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No traffic data yet. Click ▶ Start Demo in the sidebar.")

    # Trigger a visible effect for the most recent critical alert
    if alerts:
        latest = alerts[-1]
        cls = latest.get("label", "benign")
        sev = latest.get("severity", SEVERITY_MAP.get(cls, ("none", "🟢"))[0])
        icon = SEVERITY_MAP.get(cls, ("none", "🟢"))[1]
        if sev == "critical":
            st.toast(f"{icon} Critical threat detected: {cls.upper()}", icon="🚨")

# ------------------------------------------------------------
# TAB 2 — Analytics
# ------------------------------------------------------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        donut = go.Figure(
            data=[go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.55,
                marker=dict(colors=["#2ecc71", "#ff4b4b", "#e74c3c", "#ffc107"]),
            )]
        )
        donut.update_layout(
            paper_bgcolor="#0e1117",
            font_color="white",
            height=350,
            showlegend=True,
        )
        st.plotly_chart(donut, use_container_width=True)

    with col2:
        st.subheader("Model Performance")
        perf_df = pd.DataFrame({
            "Model": ["Random Forest", "XGBoost", "Ensemble"],
            "Accuracy (%)": [99.99, 100.00, 100.00],
        })
        fig_bar = px.bar(
            perf_df, x="Model", y="Accuracy (%)",
            color="Model",
            color_discrete_sequence=["#3498db", "#9b59b6", "#2ecc71"],
        )
        fig_bar.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="white",
            height=350,
            showlegend=False,
        )
        fig_bar.update_yaxes(range=[99.9, 100.05])
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Confusion Matrix (Ensemble, 120k test rows)")
    conf_matrix = [[29998, 1, 1, 0], [0, 29999, 1, 0], [0, 0, 30000, 0], [0, 0, 0, 30000]]
    labels = ["benign", "ddos", "malware", "portscan"]
    fig_cm = px.imshow(
        conf_matrix,
        x=labels, y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig_cm.update_layout(paper_bgcolor="#0e1117", font_color="white", height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

# ------------------------------------------------------------
# TAB 3 — Alerts
# ------------------------------------------------------------
with tab3:
    st.subheader("Recent Alerts")

    filtered = [a for a in alerts if a.get("label", "benign") in severity_filter]

    if not filtered:
        st.info("No alerts match the current filter.")
    else:
        for alert in reversed(filtered[-30:]):
            cls = alert.get("label", "benign")
            sev = alert.get("severity", SEVERITY_MAP.get(cls, ("none", "🟢"))[0])
            icon = SEVERITY_MAP.get(cls, ("none", "🟢"))[1]
            css_class = "critical-alert" if sev == "critical" else (
                "high-alert" if sev == "high" else "benign-alert"
            )
            ts = alert.get("timestamp", "—")
            src = alert.get("src_ip", "—")
            dst = alert.get("dst_ip", "—")
            conf = alert.get("confidence", None)
            conf_str = f"{conf:.1%}" if isinstance(conf, (int, float)) else "—"
            st.markdown(
                f'<div class="{css_class}">{icon} <b>{cls.upper()}</b> '
                f'&nbsp;|&nbsp; {ts} &nbsp;|&nbsp; {src} → {dst} '
                f'&nbsp;|&nbsp; confidence: {conf_str}</div>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------
# TAB 4 — AI Assistant
# ------------------------------------------------------------
with tab4:
    st.subheader("🤖 Ask the Security Assistant")
    st.caption("Powered by a locally hosted LLM (Ollama / tinyllama)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_q = st.chat_input("Ask about current threats, e.g. 'Why was this flagged as DDoS?'")
    if user_q:
        st.session_state.chat_history.append(("user", user_q))
        with st.chat_message("user"):
            st.write(user_q)
        # Replace this with your actual step3_ai.py / Ollama call
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = "AI explanation layer not connected in this preview — hook this up to your step3_ai.py Ollama call."
                st.write(response)
        st.session_state.chat_history.append(("assistant", response))

# ============================================================
# AUTO REFRESH
# ============================================================
if auto_refresh:
    time.sleep(5)
    st.rerun()
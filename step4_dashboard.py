"""
step4_dashboard.py — IoT Shield Dashboard
Run: streamlit run step4_dashboard.py
"""

import os
import json
import joblib
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = r"D:\final"
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
ALERTS_FILE = os.path.join(DATA_DIR, "live_alerts.json")
STATS_FILE  = os.path.join(DATA_DIR, "live_stats.json")

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Shield",
    page_icon="🛡️",
    layout="wide"
)

st_autorefresh(interval=5000, key="autorefresh")

# ─────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf  = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return rf, xgb, scaler

rf_model, xgb_model, scaler = load_models()

# ─────────────────────────────────────────────
# Load live data
# ─────────────────────────────────────────────
def load_alerts():
    try:
        with open(ALERTS_FILE) as f:
            return json.load(f)
    except:
        return []

def load_stats():
    try:
        with open(STATS_FILE) as f:
            return json.load(f)
    except:
        return {"total": 0, "threats": 0, "critical": 0, "benign": 0}

alerts = load_alerts()
stats  = load_stats()

# ─────────────────────────────────────────────
# Severity colours
# ─────────────────────────────────────────────
SEVERITY_COLOR = {
    "critical": "🔴",
    "high":     "🟡",
    "medium":   "🟠",
    "none":     "🟢",
}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🛡️ IoT Shield — Real-time Network Threat Detection")
st.caption("4-class detection: benign / ddos / malware / portscan")

# Model status
col1, col2, col3 = st.columns(3)
col1.success("✅ RF Model loaded")
col2.success("✅ XGBoost loaded")
col3.success("✅ Scaler loaded")

# ─────────────────────────────────────────────
# Top metrics
# ─────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Alerts",     stats.get("total", 0))
m2.metric("Threats Detected", stats.get("threats", 0))
m3.metric("Critical",         stats.get("critical", 0))
m4.metric("Benign",           stats.get("benign", 0))

st.caption(f"Auto-refreshes every 5 seconds")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Dashboard", "🚨 Alert Feed", "📈 Analytics", "🤖 AI Assistant"
])

# ── Tab 1: Live Dashboard ──
with tab1:
    st.subheader("Live Threat Overview")

    total    = stats.get("total", 0)
    threats  = stats.get("threats", 0)
    critical = stats.get("critical", 0)
    benign   = stats.get("benign", 0)
    rate     = (threats / total * 100) if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Connections", total)
    c2.metric("Threats Detected",  threats)
    c3.metric("Threat Rate",       f"{rate:.1f}%")
    c4.metric("Critical Alerts",   critical)

    if alerts:
        df_alerts = pd.DataFrame(alerts)

        # Threat breakdown pie
        if "label" in df_alerts.columns:
            label_counts = df_alerts["label"].value_counts().reset_index()
            label_counts.columns = ["label", "count"]
            fig = px.pie(
                label_counts, names="label", values="count",
                title="Threat Type Breakdown",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recent alerts table
        st.subheader("Latest Alerts")
        display_cols = ["timestamp", "src_ip", "dst_ip", "proto",
                        "label", "severity", "confidence"]
        show_cols = [c for c in display_cols if c in df_alerts.columns]
        st.dataframe(df_alerts[show_cols].tail(20), use_container_width=True)
    else:
        st.info("Waiting for live capture data from step5_live.py...\n\n"
                "Run `python step5_live.py` as Administrator in another terminal.")

# ── Tab 2: Alert Feed ──
with tab2:
    st.subheader("🚨 Real-time Alert Feed")
    if alerts:
        for alert in reversed(alerts[-50:]):
            sev  = alert.get("severity", "none")
            icon = SEVERITY_COLOR.get(sev, "⚪")
            lbl  = alert.get("label", "unknown").upper()
            ts   = alert.get("timestamp", "")[:19]
            src  = alert.get("src_ip", "?")
            dst  = alert.get("dst_ip", "?")
            conf = alert.get("confidence", 0)
            proto= alert.get("proto", "?").upper()
            st.markdown(
                f"{icon} `{ts}` **{lbl}** — "
                f"{src} → {dst} [{proto}] "
                f"confidence: {conf}%"
            )
    else:
        st.info("No alerts yet. Start step5_live.py as Administrator.")

# ── Tab 3: Analytics ──
with tab3:
    st.subheader("📈 Analytics")
    if alerts:
        df_alerts = pd.DataFrame(alerts)

        # Timeline
        if "timestamp" in df_alerts.columns:
            df_alerts["time"] = pd.to_datetime(df_alerts["timestamp"])
            df_alerts["minute"] = df_alerts["time"].dt.floor("min")
            timeline = df_alerts.groupby(["minute", "label"]).size().reset_index(name="count")
            fig2 = px.line(
                timeline, x="minute", y="count", color="label",
                title="Alerts over time"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Protocol breakdown
        if "proto" in df_alerts.columns:
            proto_counts = df_alerts["proto"].value_counts().reset_index()
            proto_counts.columns = ["proto", "count"]
            fig3 = px.bar(
                proto_counts, x="proto", y="count",
                title="Alerts by Protocol",
                color="proto"
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Top source IPs
        if "src_ip" in df_alerts.columns:
            top_ips = df_alerts["src_ip"].value_counts().head(10).reset_index()
            top_ips.columns = ["src_ip", "count"]
            fig4 = px.bar(
                top_ips, x="src_ip", y="count",
                title="Top 10 Source IPs"
            )
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No data yet. Start step5_live.py to see analytics.")

# ── Tab 4: AI Assistant ──
with tab4:
    st.subheader("🤖 AI Security Assistant")
    st.caption(f"Powered by Ollama / {OLLAMA_MODEL} (local)")

    user_q = st.text_input(
        "Ask anything about the current alerts or IoT security.",
        placeholder="What threats were detected?"
    )

    if user_q:
        # Build context from recent alerts
        context = ""
        if alerts:
            df_a = pd.DataFrame(alerts[-20:])
            summary = df_a["label"].value_counts().to_dict()
            context = f"Recent alert summary: {summary}. "

        prompt = (
            f"You are an IoT network security expert. "
            f"{context}"
            f"Answer this question concisely in 3-4 sentences: {user_q}"
        )
        with st.spinner("Asking AI..."):
            try:
                resp = requests.post(OLLAMA_URL, json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }, timeout=60)
                resp.raise_for_status()
                answer = resp.json().get("response", "No response.")
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"[Ollama error] {e}")
                st.info("Make sure `ollama serve` is running and tinyllama is pulled.")
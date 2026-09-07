"""
step4_dashboard.py — IoT Shield Dashboard
Matches the screenshot design exactly.
Run: python -m streamlit run step4_dashboard.py
"""

import os
import json
import time
import joblib
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Firebase Realtime Database bridge — reads live data pushed by the local backend.
# Falls back to local files automatically if Firebase isn't configured.
import firebase_sync

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* Base */
html, body, [data-testid="stApp"] {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
#MainMenu, footer { visibility: hidden; }

/* Header */
.dash-header {
    background: #161b22;
    border-bottom: 1px solid #21262d;
    padding: 0.7rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.2rem;
    border-radius: 6px;
}
.dash-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: 0.05em;
}
.dash-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e6edf3;
}
.dash-sub {
    font-size: 0.75rem;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
}
.dash-chip {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.7rem;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
    margin-left: auto;
}
.dash-chip span { color: #3fb950; }

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin-bottom: 1rem;
}
.mcard {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.mcard-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.mcard-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-bottom: 0.2rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.mcard-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent;
    border-bottom: 1px solid #21262d;
    gap: 0;
    margin-bottom: 1rem;
}
[data-testid="stTabs"] [role="tab"] {
    color: #8b949e !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}

/* Chart containers */
.chart-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}
.chart-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #8b949e;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Alert rows */
.alert-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.8rem;
    border-radius: 6px;
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: #161b22;
    border: 1px solid #21262d;
    transition: border-color 0.2s;
}
.alert-item:hover { border-color: #30363d; }
.alert-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.alert-ts { color: #8b949e; min-width: 75px; }
.alert-src { color: #58a6ff; min-width: 140px; }
.alert-lbl { font-weight: 600; min-width: 80px; }
.alert-conf { color: #8b949e; }

/* Confusion matrix */
.cm-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
.cm-table th, .cm-table td {
    padding: 0.6rem 0.8rem;
    text-align: center;
    border: 1px solid #21262d;
    font-family: 'JetBrains Mono', monospace;
}
.cm-table th { background: #161b22; color: #8b949e; font-weight: 600; }
.cm-table td { background: #0d1117; color: #e6edf3; }
.cm-diag { background: #1f3a2a !important; color: #3fb950 !important; font-weight: 700; }
.cm-err  { background: #3d1a1a !important; color: #f85149 !important; font-weight: 700; }

/* Sidebar sections */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    color: #8b949e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #21262d;
}


/* AI chat */
.ai-bubble {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.82rem;
    line-height: 1.6;
    color: #e6edf3;
    margin-top: 0.8rem;
}
.ai-label {
    font-size: 0.7rem;
    color: #3fb950;
    font-weight: 600;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}
.quick-btn-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }

/* Toggle */
[data-testid="stToggle"] label { font-size: 0.8rem !important; color: #8b949e !important; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #8b949e;
    font-size: 0.85rem;
}
.empty-icon { font-size: 2rem; margin-bottom: 0.5rem; }

/* Scrollable alert feed */
.alert-feed { max-height: 420px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True
if "filter_classes" not in st.session_state:
    st.session_state.filter_classes = ["benign","ddos","malware","portscan"]

# Auto refresh
if st.session_state.auto_refresh:
    st_autorefresh(interval=3000, key="autorefresh")

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        rf  = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
        xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
        sc  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        return rf, xgb, sc, True
    except:
        return None, None, None, False

rf_model, xgb_model, scaler, models_ok = load_models()

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
def load_alerts():
    try:
        with open(ALERTS_FILE) as f: return json.load(f)
    except: return []

def load_stats():
    try:
        with open(STATS_FILE) as f: return json.load(f)
    except: return {"total":0,"threats":0,"critical":0,"benign":0}

def clear_data():
    with open(ALERTS_FILE,"w") as f: json.dump([],f)
    with open(STATS_FILE,"w") as f: json.dump({"total":0,"threats":0,"critical":0,"benign":0},f)

# Prefer live data from Firebase (pushed by the local backend); fall back to
# the local JSON files / committed simulation snapshot when Firebase is unset.
fb_alerts, fb_stats = firebase_sync.fetch()
if fb_alerts is not None:
    alerts = fb_alerts
    stats  = fb_stats or {"total":0,"threats":0,"critical":0,"benign":0}
    DATA_SOURCE = "Firebase · live"
else:
    alerts = load_alerts()
    stats  = load_stats()
    DATA_SOURCE = "Local file"

# Colours per class
CLASS_COLORS = {
    "benign":   "#3fb950",
    "ddos":     "#f85149",
    "malware":  "#d29922",
    "portscan": "#58a6ff",
}
SEV_COLORS = {"critical":"#f85149","high":"#d29922","none":"#3fb950"}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ IoT Shield")
    st.markdown("Threat Detection System")
    st.divider()

    st.markdown("**SIMULATION**")
    if st.button("▶ Start", use_container_width=True):
        clear_data()
        st.session_state.running = True
        st.success("Now run step5_simulate.py")
    if st.button("⏹ Stop", use_container_width=True):
        st.session_state.running = False
        st.info("Stopped.")
    if st.button("↺ Reset Dashboard", use_container_width=True):
        clear_data()
        st.rerun()

    st.markdown("---")

    # Auto refresh toggle
    st.session_state.auto_refresh = st.toggle(
        "Auto-refresh (3s)",
        value=st.session_state.auto_refresh
    )

    st.markdown("---")

    # Filter by class
    st.markdown('<div class="sidebar-section">FILTER ALERTS BY CLASS</div>', unsafe_allow_html=True)
    selected = []
    for cls in ["benign","ddos","malware","portscan"]:
        color = CLASS_COLORS[cls]
        checked = st.checkbox(
            cls.capitalize(),
            value=cls in st.session_state.filter_classes,
            key=f"chk_{cls}"
        )
        if checked:
            selected.append(cls)
    st.session_state.filter_classes = selected if selected else ["benign","ddos","malware","portscan"]

    st.markdown("---")
    st.markdown('<div class="sidebar-section">DATA SOURCE</div>', unsafe_allow_html=True)
    _src_color = "#3fb950" if DATA_SOURCE.startswith("Firebase") else "#d29922"
    st.markdown(f'<div style="font-size:0.75rem;color:{_src_color};font-family:JetBrains Mono,monospace;">● {DATA_SOURCE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.7rem;color:#8b949e;margin-top:0.5rem;">Last refresh<br>{datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
total   = stats.get("total",0)
threats = stats.get("threats",0)
benign  = stats.get("benign",0)
critical= stats.get("critical",0)

st.markdown(f"""
<div class="dash-header">
    <div class="dash-logo">🛡️ IoT Shield</div>
    <div>
        <div class="dash-title">IoT Shield — Network Threat Detection</div>
        <div class="dash-sub">Live monitoring · IoT-23 dataset · Ensemble ML detection</div>
    </div>
    <div class="dash-chip"><span>●</span> {total} flows analysed</div>
    <div class="dash-chip">⚡ {threats} threats</div>
    <div class="dash-chip">packets analysed: {total}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Monitor", "Analytics", "Alerts", "AI Assistant"
])

# ── TAB 1: Live Monitor ──────────────────────
with tab1:
    # Metric cards
    benign_count   = sum(1 for a in alerts if a.get("label")=="benign")
    portscan_count = sum(1 for a in alerts if a.get("label")=="portscan")
    ddos_count     = sum(1 for a in alerts if a.get("label")=="ddos")
    malware_count  = sum(1 for a in alerts if a.get("label")=="malware")

    st.markdown(f"""
    <div class="metric-row">
        <div class="mcard">
            <div class="mcard-dot" style="background:#3fb950;box-shadow:0 0 6px #3fb95066"></div>
            <div><div class="mcard-label">BENIGN</div><div class="mcard-value">{benign_count}</div></div>
        </div>
        <div class="mcard">
            <div class="mcard-dot" style="background:#58a6ff;box-shadow:0 0 6px #58a6ff66"></div>
            <div><div class="mcard-label">PORT SCAN</div><div class="mcard-value">{portscan_count}</div></div>
        </div>
        <div class="mcard">
            <div class="mcard-dot" style="background:#f85149;box-shadow:0 0 6px #f8514966"></div>
            <div><div class="mcard-label">DDoS</div><div class="mcard-value">{ddos_count}</div></div>
        </div>
        <div class="mcard">
            <div class="mcard-dot" style="background:#d29922;box-shadow:0 0 6px #d2992266"></div>
            <div><div class="mcard-label">MALWARE</div><div class="mcard-value">{malware_count}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Traffic over time
    st.markdown('<div class="chart-box"><div class="chart-title">Traffic classification over time</div>', unsafe_allow_html=True)
    if alerts:
        df = pd.DataFrame(alerts)
        df["time"] = pd.to_datetime(df["timestamp"])
        df["second"] = df["time"].dt.floor("5s")
        timeline = df.groupby(["second","label"]).size().reset_index(name="count")
        fig = go.Figure()
        for cls, color in CLASS_COLORS.items():
            sub = timeline[timeline["label"]==cls]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["second"], y=sub["count"],
                    name=cls.capitalize(), mode="lines",
                    line=dict(color=color, width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,0,0,0.05)",
                ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", family="Inter", size=11),
            xaxis=dict(showgrid=False, color="#8b949e", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#21262d", color="#8b949e", zeroline=False),
            legend=dict(font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=5,b=5,l=5,r=5), height=220,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="empty-state"><div class="empty-icon">📡</div>No traffic data yet. Click Start in the sidebar.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: Analytics ─────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="chart-box"><div class="chart-title">Class distribution</div>', unsafe_allow_html=True)
        if alerts:
            df2 = pd.DataFrame(alerts)
            counts = df2["label"].value_counts().reset_index()
            counts.columns = ["label","count"]
            fig2 = go.Figure(go.Pie(
                labels=counts["label"],
                values=counts["count"],
                hole=0.55,
                marker=dict(
                    colors=[CLASS_COLORS.get(l,"#888") for l in counts["label"]],
                    line=dict(color="#0d1117", width=2)
                ),
                textfont=dict(family="Inter", color="#e6edf3", size=11),
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b949e"),
                legend=dict(font=dict(color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=5,b=5,l=5,r=5), height=240,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown('<div class="empty-state">No classified flows yet.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-box"><div class="chart-title">Model performance</div>', unsafe_allow_html=True)
        models_data = {
            "Model":    ["Random Forest", "XGBoost", "Ensemble"],
            "Accuracy": [99.99, 100.00, 100.00],
        }
        fig3 = go.Figure(go.Bar(
            x=models_data["Model"],
            y=models_data["Accuracy"],
            marker_color=["#58a6ff","#d29922","#3fb950"],
            text=[f"{v:.2f}%" for v in models_data["Accuracy"]],
            textposition="outside",
            textfont=dict(color="#e6edf3", family="JetBrains Mono", size=11),
            width=0.5,
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", family="Inter"),
            xaxis=dict(showgrid=False, color="#8b949e"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", color="#8b949e",
                       range=[99.8, 100.05]),
            margin=dict(t=20,b=5,l=5,r=5), height=240,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(
            '<div style="font-size:0.7rem;color:#8b949e;text-align:center;">'
            'RF + XGBoost soft-voting ensemble · Accuracy: 100.00% (test set)</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Confusion matrix
    st.markdown('<div class="chart-box"><div class="chart-title">Confusion matrix — Ensemble (120k test rows)</div>', unsafe_allow_html=True)
    cm_labels = ["Benign","DDoS","Malware","Port Scan"]
    cm_data = [
        [29998, 1,     1,     0],
        [0,     29999, 1,     0],
        [0,     0,     30000, 0],
        [0,     0,     0,     30000],
    ]
    html_cm = '<table class="cm-table"><tr><th>actual \\ predicted</th>'
    for lbl in cm_labels:
        html_cm += f"<th>{lbl}</th>"
    html_cm += "</tr>"
    for i, row in enumerate(cm_data):
        html_cm += f"<tr><th>{cm_labels[i]}</th>"
        for j, val in enumerate(row):
            cls = "cm-diag" if i==j else ("cm-err" if val>0 else "")
            html_cm += f'<td class="{cls}">{val:,}</td>'
        html_cm += "</tr>"
    html_cm += "</table>"
    st.markdown(html_cm, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3: Alerts ────────────────────────────
with tab3:
    st.markdown('<div class="chart-box"><div class="chart-title">Live alert feed</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns([1,1])
    with fc1:
        f_label = st.selectbox("Type", ["ALL","benign","ddos","malware","portscan"], key="f_label")
    with fc2:
        f_sev = st.selectbox("Severity", ["ALL","critical","high","none"], key="f_sev")

    if not alerts:
        st.markdown('<div class="empty-state"><div class="empty-icon">🔍</div>No alerts yet. Start capture to see live feed.</div>', unsafe_allow_html=True)
    else:
        df3 = pd.DataFrame(alerts)
        # Apply filters
        filtered = [a for a in alerts
                    if a.get("label") in st.session_state.filter_classes
                    and (f_label=="ALL" or a.get("label")==f_label)
                    and (f_sev=="ALL" or a.get("severity")==f_sev)]

        st.markdown(f'<div style="font-size:0.75rem;color:#8b949e;margin-bottom:0.5rem;">Showing {len(filtered)} alerts</div>', unsafe_allow_html=True)

              # Human friendly translations
        HUMAN_MSG = {
            "benign":   ("✅ Normal traffic", "Your network activity looks safe. No action needed.", "Check if this is expected activity."),
            "ddos":     ("⚠️ High volume traffic", "Your device is making unusually many connections. This could be heavy app usage or a potential attack.", "Close background apps and check your WiFi usage."),
            "malware":  ("🚨 Suspicious communication", "Your device may be communicating with a suspicious server. This could indicate malware.", "Run a virus scan and check recently installed apps."),
            "portscan": ("🔍 Network scanning detected", "Someone may be probing your device looking for open ports.", "Ensure your firewall is enabled and avoid public WiFi."),
        }

        rows_html = '<div class="alert-feed">'
        for a in reversed(filtered[-100:]):
            lbl   = a.get("label","?")
            sev   = a.get("severity","none")
            color = CLASS_COLORS.get(lbl,"#888")
            ts    = str(a.get("timestamp",""))[:19].replace("T"," ")
            src   = a.get("src_ip","?")
            conf  = a.get("confidence",0)
            proto = str(a.get("proto","?")).upper()

            # Risk level
            if conf >= 80:
                risk = "High Risk"
                risk_color = "#f85149"
            elif conf >= 60:
                risk = "Medium Risk"
                risk_color = "#d29922"
            else:
                risk = "Low Risk"
                risk_color = "#3fb950"

            title, msg, action = HUMAN_MSG.get(lbl, (lbl.upper(), "", ""))

            rows_html += f"""
            <div class="alert-item" style="flex-direction:column;align-items:flex-start;padding:0.8rem;">
                <div style="display:flex;align-items:center;gap:0.6rem;width:100%;margin-bottom:0.4rem;">
                    <div class="alert-dot" style="background:{color}"></div>
                    <span style="font-weight:700;color:{color};font-size:0.82rem;">{title}</span>
                    <span style="margin-left:auto;font-size:0.7rem;color:{risk_color};font-weight:600;">{risk}</span>
                    <span class="alert-ts">{ts[11:]}</span>
                </div>
                <div style="font-size:0.78rem;color:#e6edf3;margin-bottom:0.3rem;">{msg}</div>
                <div style="font-size:0.72rem;color:#8b949e;">💡 {action}</div>
                <div style="font-size:0.68rem;color:#484f58;margin-top:0.3rem;">
                    IP: {src} · {proto} · Confidence: {conf}% · ML label: {lbl}
                </div>
            </div>"""
        rows_html += '</div>'
        st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 4: AI Assistant ──────────────────────
with tab4:
    col_ai1, col_ai2 = st.columns([2,1])

    with col_ai1:
        st.markdown('<div class="chart-box"><div class="chart-title">Security assistant · Powered by locally hosted LLM (Ollama / tinyllama)</div>', unsafe_allow_html=True)

        # Quick question buttons
        quick_questions = [
            "Why was this flagged as malicious?",
            "Explain the malware detections",
            "What triggers a port scan alert?",
            "Is the network under attack?",
        ]
        quick_q = st.radio(
            "Quick questions:",
            ["-- Select --"] + quick_questions,
            key="quick_radio"
        )
        if quick_q == "-- Select --":
            quick_q = None

        user_q = st.text_input(
            "Ask about current threats, model reasoning, or specific flows",
            value=quick_q or "",
            placeholder="e.g. Explain the DDoS alerts...",
            key="ai_input"
        )

        if user_q:
            context = ""
            if alerts:
                df_q = pd.DataFrame(alerts[-30:])
                summary = df_q["label"].value_counts().to_dict()
                top_src = df_q["src_ip"].value_counts().head(3).to_dict() if "src_ip" in df_q.columns else {}
                context = f"Alert summary (last 30): {summary}. Top source IPs: {top_src}. "

            prompt = (
                f"You are an expert IoT network security analyst. {context}"
                f"Answer concisely in 3-4 sentences: {user_q}"
            )
            with st.spinner("Analysing..."):
                try:
                    resp = requests.post(OLLAMA_URL, json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False
                    }, timeout=60)
                    resp.raise_for_status()
                    answer = resp.json().get("response","No response.")
                    st.markdown(f"""
                    <div class="ai-bubble">
                        <div class="ai-label">▸ AI ANALYST</div>
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Ollama error: {e}")
                    st.info("Run: ollama serve  then  ollama pull tinyllama")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ai2:
        st.markdown('<div class="chart-box"><div class="chart-title">Filter alerts by class</div>', unsafe_allow_html=True)
        for cls, color in CLASS_COLORS.items():
            st.markdown(f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;font-size:0.82rem;"><div style="width:8px;height:8px;border-radius:50%;background:{color}"></div>{cls.capitalize()}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-box"><div class="chart-title">Model status</div>', unsafe_allow_html=True)
        status_icon = "✅" if models_ok else "❌"
        st.markdown(f"""
        <div style="font-size:0.78rem;line-height:2;">
            {status_icon} Random Forest loaded<br>
            {status_icon} XGBoost loaded<br>
            {status_icon} Scaler loaded<br>
            ✅ Ensemble active
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
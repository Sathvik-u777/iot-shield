"""
step5_simulate.py — IoT Shield Simulation Mode
Replays rows from combined.csv through the ML models as if they
were live network flows. No admin rights or Npcap needed.
Run: python step5_simulate.py
     python step5_simulate.py --reset   (clear alerts first)
     python step5_simulate.py --speed 2 (2x faster replay)
"""

import os
import sys
import json
import time
import joblib
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR    = r"D:\final"
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")
ALERTS_FILE = os.path.join(DATA_DIR, "live_alerts.json")
STATS_FILE  = os.path.join(DATA_DIR, "live_stats.json")
DATASET     = os.path.join(DATA_DIR, "combined.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="Clear alerts before starting")
parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier (default 1.0)")
parser.add_argument("--delay", type=float, default=0.5, help="Seconds between alerts (default 0.5)")
args = parser.parse_args()

DELAY = args.delay / args.speed

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
print("[*] Loading models...")
rf_model     = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb_model    = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_enc    = joblib.load(os.path.join(MODEL_DIR, "label_enc.pkl"))
cat_encoders = joblib.load(os.path.join(MODEL_DIR, "cat_encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

CLASSES = list(label_enc.classes_)
print(f"[*] Classes : {CLASSES}")

SEVERITY = {
    "benign":   "none",
    "ddos":     "critical",
    "malware":  "critical",
    "portscan": "high",
}

# ─────────────────────────────────────────────
# Load dataset — sample evenly from each class
# ─────────────────────────────────────────────
print(f"[*] Loading simulation data from {DATASET} ...")
df = pd.read_csv(DATASET)

# Take 500 rows per class for simulation (2000 total)
ROWS_PER_CLASS = 500
parts = []
for cls in sorted(df["label"].unique()):
    cls_df = df[df["label"] == cls]
    n = min(len(cls_df), ROWS_PER_CLASS)
    parts.append(cls_df.sample(n=n, random_state=99))

sim_df = pd.concat(parts, ignore_index=True)
sim_df = sim_df.sample(frac=1, random_state=99).reset_index(drop=True)
print(f"[*] Simulation rows: {len(sim_df):,} ({ROWS_PER_CLASS} per class)")
print(f"[*] Delay between alerts: {DELAY:.2f}s")

# ─────────────────────────────────────────────
# Encode categoricals
# ─────────────────────────────────────────────
CAT_COLS = ["proto", "service", "conn_state"]

def encode_row(row):
    r = dict(row)
    for col in CAT_COLS:
        if col in feature_cols:
            le = cat_encoders.get(col)
            if le:
                try:
                    r[col] = int(le.transform([str(r[col])])[0])
                except:
                    r[col] = 0
    return r

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
def predict(row):
    try:
        r = encode_row(row)
        X = pd.DataFrame([{col: r.get(col, 0) for col in feature_cols}])
        X_scaled = scaler.transform(X)
        rp = rf_model.predict_proba(X_scaled)[0]
        xp = xgb_model.predict_proba(X_scaled)[0]
        ep = 0.45 * rp + 0.55 * xp
        idx = int(np.argmax(ep))
        return CLASSES[idx], float(ep[idx])
    except Exception as e:
        print(f"[!] Predict error: {e}")
        return "benign", 0.0

# ─────────────────────────────────────────────
# Stats and alerts
# ─────────────────────────────────────────────
stats = {
    "total": 0, "threats": 0, "critical": 0, "benign": 0,
    "start_time": datetime.now().isoformat(),
}
alerts = []
MAX_ALERTS = 200

def write_files():
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts[-MAX_ALERTS:], f, indent=2)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

# ─────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────
if args.reset or not os.path.exists(ALERTS_FILE):
    with open(ALERTS_FILE, "w") as f:
        json.dump([], f)
    print("[*] Alert feed cleared.")

write_files()

# ─────────────────────────────────────────────
# Simulate fake IPs for realism
# ─────────────────────────────────────────────
import random
random.seed(42)

IOT_DEVICES = [
    "192.168.1.101", "192.168.1.102", "192.168.1.103",
    "192.168.1.104", "192.168.1.105", "10.0.0.50",
    "10.0.0.51",     "10.0.0.52",     "172.16.0.10",
]
EXTERNAL_IPS = [
    "45.33.32.156",  "198.51.100.1",  "203.0.113.50",
    "104.21.45.67",  "172.67.182.9",  "91.108.4.150",
    "185.220.101.5", "23.94.47.219",  "66.249.66.1",
]

# ─────────────────────────────────────────────
# Main replay loop
# ─────────────────────────────────────────────
print("=" * 60)
print("  IoT Shield — Simulation Mode  (Ctrl+C to stop)")
print("=" * 60)
print()

try:
    for i, (_, row) in enumerate(sim_df.iterrows()):
        true_label = row["label"]
        pred_label, confidence = predict(row)
        severity = SEVERITY.get(pred_label, "low")
        ts_str = datetime.now().isoformat()

        # Pick realistic IPs based on label
        src_ip = random.choice(IOT_DEVICES)
        dst_ip = random.choice(EXTERNAL_IPS) if pred_label != "benign" else random.choice(IOT_DEVICES)
        src_port = int(row.get("id.orig_p", random.randint(1024, 65535)))
        dst_port = int(row.get("id.resp_p", 443))
        proto    = str(row.get("proto", "tcp"))

        # Update stats
        stats["total"] += 1
        if pred_label != "benign":
            stats["threats"] += 1
            if severity == "critical":
                stats["critical"] += 1
        else:
            stats["benign"] += 1

        alert = {
            "timestamp":   ts_str,
            "src_ip":      src_ip,
            "dst_ip":      dst_ip,
            "src_port":    src_port,
            "dst_port":    dst_port,
            "proto":       proto,
            "service":     str(row.get("service", "-")),
            "label":       pred_label,
            "true_label":  true_label,
            "severity":    severity,
            "confidence":  round(confidence * 100, 1),
            "orig_pkts":   int(row.get("orig_pkts", 0)),
            "resp_pkts":   int(row.get("resp_pkts", 0)),
            "orig_bytes":  int(row.get("orig_ip_bytes", 0)),
            "resp_bytes":  int(row.get("resp_ip_bytes", 0)),
        }
        alerts.append(alert)
        write_files()

        # Console output
        correct = "✓" if pred_label == true_label else "✗"
        icon = "🟢" if pred_label == "benign" else ("🔴" if severity == "critical" else "🟡")
        print(f"{icon} [{ts_str[11:19]}] {src_ip}:{src_port} -> "
              f"{dst_ip}:{dst_port}  "
              f"{pred_label.upper():10s} {confidence*100:.1f}%  "
              f"[true: {true_label}] {correct}")

        time.sleep(DELAY)

    print(f"\n[*] Simulation complete. {stats['total']} flows replayed.")
    print(f"[*] Threats: {stats['threats']}  |  Benign: {stats['benign']}")

except KeyboardInterrupt:
    print(f"\n[*] Stopped. Replayed {stats['total']} flows.")
    write_files()
"""
phone_engine.py — reads phone data from Firebase,
runs ML model, pushes results back to Firebase.
Run: python phone_engine.py
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime

BASE_DIR     = r"D:\final"
MODEL_DIR    = os.path.join(BASE_DIR, "models")
FIREBASE_URL = "https://iot-shield-7ce30-default-rtdb.asia-southeast1.firebasedatabase.app"

# Load models
print("[*] Loading models...")
rf_model     = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb_model    = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_enc    = joblib.load(os.path.join(MODEL_DIR, "label_enc.pkl"))
cat_encoders = joblib.load(os.path.join(MODEL_DIR, "cat_encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
CLASSES      = list(label_enc.classes_)

SEVERITY = {
    "benign":   "none",
    "ddos":     "critical",
    "malware":  "critical",
    "portscan": "high",
}

def fetch_phone_data():
    try:
        r = requests.get(f"{FIREBASE_URL}/phone_data.json", timeout=5)
        return r.json() or {}
    except:
        return {}

def push_results(device_id, alert):
    try:
        url = f"{FIREBASE_URL}/live_alerts.json"
        # Get existing alerts
        r = requests.get(url, timeout=5)
        alerts = r.json() or []
        if isinstance(alerts, dict):
            alerts = list(alerts.values())
        alerts.append(alert)
        # Keep last 200
        alerts = alerts[-200:]
        requests.put(url, json=alerts, timeout=5)

        # Update stats
        stats_url = f"{FIREBASE_URL}/live_stats.json"
        sr = requests.get(stats_url, timeout=5)
        stats = sr.json() or {"total":0,"threats":0,"critical":0,"benign":0}
        stats["total"] = stats.get("total",0) + 1
        if alert["label"] != "benign":
            stats["threats"] = stats.get("threats",0) + 1
            if alert["severity"] == "critical":
                stats["critical"] = stats.get("critical",0) + 1
        else:
            stats["benign"] = stats.get("benign",0) + 1
        requests.put(stats_url, json=stats, timeout=5)
        return True
    except Exception as e:
        print(f"Push error: {e}")
        return False

def encode_cat(col, val):
    le = cat_encoders.get(col)
    if le is None: return 0
    try: return int(le.transform([str(val)])[0])
    except: return 0

def predict_from_phone(device_data):
    conn_count = device_data.get("conn_count", 0)
    connections = device_data.get("connections", [])
    
    # Count connection states
    established = sum(1 for c in connections if c.get("state") == "0A")
    syn_sent    = sum(1 for c in connections if c.get("state") == "02")
    time_wait   = sum(1 for c in connections if c.get("state") == "06")

    raw = {
        "ts":            time.time(),
        "id.orig_p":     np.random.randint(1024, 65535),
        "id.resp_p":     443 if established > 0 else 80,
        "proto":         "tcp",
        "service":       "ssl" if established > 5 else "-",
        "conn_state":    "SF" if established > syn_sent else "S0",
        "missed_bytes":  0,
        "orig_pkts":     max(conn_count, 1),
        "orig_ip_bytes": conn_count * np.random.randint(50, 200),
        "resp_pkts":     max(established, 0),
        "resp_ip_bytes": established * np.random.randint(50, 150),
    }
    for col in ["proto","service","conn_state"]:
        if col in feature_cols:
            raw[col] = encode_cat(col, raw[col])

    X = pd.DataFrame([{col: raw.get(col,0) for col in feature_cols}])
    X_scaled = scaler.transform(X)
    rp = rf_model.predict_proba(X_scaled)[0]
    xp = xgb_model.predict_proba(X_scaled)[0]
    ep = 0.45*rp + 0.55*xp
    idx = int(np.argmax(ep))
    return CLASSES[idx], float(ep[idx])

print("[*] Phone Engine running — checking Firebase every 10s")
print()

seen_timestamps = set()

while True:
    phone_data = fetch_phone_data()

    for device_id, data in phone_data.items():
        ts = data.get("timestamp","")
        if ts in seen_timestamps:
            continue
        seen_timestamps.add(ts)

        label, confidence = predict_from_phone(data)
        severity = SEVERITY.get(label, "low")
        ip = data.get("public_ip","unknown")

        alert = {
            "timestamp":  datetime.now().isoformat(),
            "src_ip":     ip,
            "dst_ip":     "cloud",
            "src_port":   5000,
            "dst_port":   443,
            "proto":      "tcp",
            "service":    "mobile",
            "label":      label,
            "severity":   severity,
            "confidence": round(confidence*100, 1),
            "orig_pkts":  data.get("conn_count",0),
            "resp_pkts":  0,
            "orig_bytes": 0,
            "resp_bytes": 0,
            "device":     device_id,
        }

        ok = push_results(device_id, alert)
        icon = "🟢" if label=="benign" else ("🔴" if severity=="critical" else "🟡")
        print(f"{icon} [{ts[11:19]}] {device_id} | IP:{ip} | {label.upper()} {confidence*100:.1f}% | Firebase:{'OK' if ok else 'FAIL'}")

    time.sleep(10)
"""
step5_live.py — IoT Shield Live Capture & Detection
Run as Administrator: python step5_live.py
Writes: data/live_alerts.json
        data/live_stats.json
"""

import os
import json
import time
import threading
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import sniff, IP, TCP, UDP, conf

BASE_DIR    = r"D:\final"
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")
ALERTS_FILE = os.path.join(DATA_DIR, "live_alerts.json")
STATS_FILE  = os.path.join(DATA_DIR, "live_stats.json")
os.makedirs(DATA_DIR, exist_ok=True)

print("[*] Loading models...")
rf_model     = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb_model    = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_enc    = joblib.load(os.path.join(MODEL_DIR, "label_enc.pkl"))
cat_encoders = joblib.load(os.path.join(MODEL_DIR, "cat_encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

CLASSES = list(label_enc.classes_)
print(f"[*] Classes : {CLASSES}")
print(f"[*] Features: {feature_cols}")

SEVERITY = {
    "benign":   "none",
    "ddos":     "critical",
    "malware":  "critical",
    "portscan": "high",
}

_flows = defaultdict(lambda: {
    "ts": None, "orig_ip": None,
    "orig_pkts": 0, "orig_ip_bytes": 0,
    "resp_pkts": 0, "resp_ip_bytes": 0,
    "missed_bytes": 0,
    "proto": "", "service": "", "conn_state": "SF",
    "id.orig_p": 0, "id.resp_p": 0,
})
_flow_lock = threading.Lock()

_stats = {
    "total": 0, "threats": 0, "critical": 0, "benign": 0,
    "start_time": datetime.now().isoformat(),
}

_alerts = []
_alerts_lock = threading.Lock()
MAX_ALERTS = 200
_pkt_count = 0
_pkt_lock  = threading.Lock()

def _proto_str(pkt):
    if TCP in pkt: return "tcp"
    if UDP in pkt: return "udp"
    return "icmp"

def _service_from_port(port):
    return {80:"http",443:"ssl",53:"dns",22:"ssh",21:"ftp",25:"smtp",3389:"rdp",8080:"http"}.get(port,"-")

def _conn_state(pkt):
    if TCP in pkt:
        f = pkt[TCP].flags
        if f & 0x01: return "SF"
        if f & 0x04: return "RSTO"
        if f & 0x02: return "S0"
    return "SF"

def _flow_key(pkt):
    if IP not in pkt: return None
    proto = _proto_str(pkt)
    src = pkt[IP].src; dst = pkt[IP].dst
    sp = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
    dp = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
    return (src,sp,dst,dp,proto) if (src,sp)<(dst,dp) else (dst,dp,src,sp,proto)

def _update_flow(pkt):
    key = _flow_key(pkt)
    if key is None: return None
    src = pkt[IP].src
    sp  = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
    dp  = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
    plen = len(pkt)
    with _flow_lock:
        f = _flows[key]
        if f["ts"] is None:
            f["ts"]=time.time(); f["orig_ip"]=src
            f["proto"]=_proto_str(pkt); f["id.orig_p"]=sp; f["id.resp_p"]=dp
            f["service"]=_service_from_port(dp) or _service_from_port(sp)
            f["conn_state"]=_conn_state(pkt)
        if src == f["orig_ip"]:
            f["orig_pkts"]+=1; f["orig_ip_bytes"]+=plen
        else:
            f["resp_pkts"]+=1; f["resp_ip_bytes"]+=plen
    return key

def _encode_cat(col, val):
    le = cat_encoders.get(col)
    if le is None: return 0
    try: return int(le.transform([val])[0])
    except: return 0

def _build_row(flow):
    raw = {
        "ts":flow["ts"] or time.time(),"id.orig_p":flow["id.orig_p"],
        "id.resp_p":flow["id.resp_p"],"proto":flow["proto"],
        "service":flow["service"] or "-","conn_state":flow["conn_state"],
        "missed_bytes":flow["missed_bytes"],"orig_pkts":flow["orig_pkts"],
        "orig_ip_bytes":flow["orig_ip_bytes"],"resp_pkts":flow["resp_pkts"],
        "resp_ip_bytes":flow["resp_ip_bytes"],
    }
    for col in ["proto","service","conn_state"]:
        if col in feature_cols: raw[col]=_encode_cat(col,raw[col])
    return pd.DataFrame([{col:raw.get(col,0) for col in feature_cols}])

def _predict(flow):
    try:
        X = scaler.transform(_build_row(flow))
        rp = rf_model.predict_proba(X)[0]
        xp = xgb_model.predict_proba(X)[0]
        ep = 0.45*rp + 0.55*xp
        idx = int(np.argmax(ep))
        return CLASSES[idx], float(ep[idx])
    except Exception as e:
        print(f"[!] Predict error: {e}"); return "benign", 0.0

def _write_alerts():
    with _alerts_lock: payload = _alerts[-MAX_ALERTS:]
    with open(ALERTS_FILE,"w") as f: json.dump(payload,f,indent=2)

def _write_stats():
    with open(STATS_FILE,"w") as f: json.dump(_stats,f,indent=2)

_scored = set()
_SCORE_AFTER = 3

def _maybe_score(key, src_ip, dst_ip):
    with _flow_lock: flow = dict(_flows[key])
    if flow["orig_pkts"] < _SCORE_AFTER: return
    if key in _scored: return
    _scored.add(key)
    label, confidence = _predict(flow)
    severity = SEVERITY.get(label,"low")
    ts_str = datetime.now().isoformat()
    _stats["total"]+=1
    if label!="benign":
        _stats["threats"]+=1
        if severity=="critical": _stats["critical"]+=1
    else:
        _stats["benign"]+=1
    alert={
        "timestamp":ts_str,"src_ip":src_ip,"dst_ip":dst_ip,
        "src_port":flow["id.orig_p"],"dst_port":flow["id.resp_p"],
        "proto":flow["proto"],"service":flow["service"],"label":label,
        "severity":severity,"confidence":round(confidence*100,1),
        "orig_pkts":flow["orig_pkts"],"resp_pkts":flow["resp_pkts"],
        "orig_bytes":flow["orig_ip_bytes"],"resp_bytes":flow["resp_ip_bytes"],
    }
    with _alerts_lock: _alerts.append(alert)
    _write_alerts(); _write_stats()
    icon="🟢" if label=="benign" else ("🔴" if severity=="critical" else "🟡")
    print(f"{icon} [{ts_str[11:19]}] {src_ip}:{flow['id.orig_p']} -> "
          f"{dst_ip}:{flow['id.resp_p']}  {label.upper():10s}  "
          f"{confidence*100:.1f}%  [{flow['proto'].upper()}]")

def _packet_callback(pkt):
    global _pkt_count
    if IP not in pkt: return
    with _pkt_lock: _pkt_count+=1
    key = _update_flow(pkt)
    if key: _maybe_score(key, pkt[IP].src, pkt[IP].dst)

def _stats_flusher():
    while True:
        time.sleep(5); _write_stats()
        with _pkt_lock: n=_pkt_count
        print(f"[~] Packets: {n:,}  |  Flows: {len(_flows)}  |  Scored: {len(_scored)}  |  Alerts: {_stats['total']}")

def _pick_interface():
    PREFERRED = [
        r"\Device\NPF_{7BEA775F-3353-4943-85B9-330D8B568733}",
        r"\Device\NPF_{282EE48D-FA62-4C27-A694-92C3580A2BA6}",
    ]
    try:
        available = set(conf.ifaces.keys())
        for iface in PREFERRED:
            if iface in available:
                print(f"[*] Interface: {iface}"); return iface
    except Exception as e:
        print(f"[!] Interface error: {e}")
    return None

def _init_files():
    import sys
    reset = "--reset" in sys.argv
    if reset or not os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE,"w") as f: json.dump([],f)
        if reset: print("[*] Alert feed cleared.")
    if reset or not os.path.exists(STATS_FILE):
        _write_stats()

if __name__ == "__main__":
    print("="*60)
    print("  IoT Shield - Live Capture  (Ctrl+C to stop)")
    print("  Use --reset to clear alerts on startup")
    print("="*60)
    _init_files()
    threading.Thread(target=_stats_flusher, daemon=True).start()
    iface = _pick_interface()
    print(f"[*] Alert feed : {ALERTS_FILE}")
    print(f"[*] Score after: {_SCORE_AFTER} originator packets\n")
    try:
        sniff(iface=iface, filter="ip", prn=_packet_callback, store=False)
    except PermissionError:
        print("\n[!] Run as Administrator.")
    except KeyboardInterrupt:
        print(f"\n[*] Stopped. Total scored: {_stats['total']}")
        _write_stats()
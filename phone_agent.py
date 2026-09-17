import requests
import time
import socket
import urllib.request
from datetime import datetime

FIREBASE_URL = "https://iot-shield-7ce30-default-rtdb.asia-southeast1.firebasedatabase.app"
DEVICE_ID = "phone_" + socket.gethostname()

def get_public_ip():
    try:
        return urllib.request.urlopen(
            "https://api.ipify.org", timeout=5
        ).read().decode()
    except:
        return "unknown"

def get_connections():
    try:
        conns = []
        for f in ["/proc/net/tcp", "/proc/net/tcp6"]:
            try:
                with open(f) as fh:
                    for line in fh.readlines()[1:]:
                        parts = line.split()
                        if len(parts) > 3:
                            conns.append({
                                "local": parts[1],
                                "remote": parts[2],
                                "state": parts[3]
                            })
            except:
                pass
        return conns[:30]
    except:
        return []

def push_to_firebase(data):
    url = f"{FIREBASE_URL}/phone_data/{DEVICE_ID}.json"
    try:
        resp = requests.put(url, json=data, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"Firebase error: {e}")
        return False

print(f"[*] IoT Shield Phone Agent")
print(f"[*] Device: {DEVICE_ID}")
print(f"[*] Ctrl+C to stop")
print()

while True:
    ip = get_public_ip()
    conns = get_connections()
    ts = datetime.now().isoformat()
    data = {
        "timestamp": ts,
        "device_id": DEVICE_ID,
        "public_ip": ip,
        "conn_count": len(conns),
        "connections": conns,
        "status": "active"
    }
    ok = push_to_firebase(data)
    print(f"[{ts[11:19]}] IP:{ip} | Conns:{len(conns)} | Firebase:{'OK' if ok else 'FAIL'}")
    time.sleep(10)
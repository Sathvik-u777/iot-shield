"""
firebase_sync.py — IoT Shield  <->  Firebase Realtime Database bridge.

WHY THIS FILE EXISTS
    Streamlit Cloud cannot run Scapy or see your LAN traffic, so the detection
    backend (step5_live.py / step5_simulate.py) has to run on your LOCAL
    machine. This module is the bridge between that local backend and the
    public cloud dashboard:

        local backend  --push-->  Firebase Realtime DB  --fetch-->  cloud dashboard

DESIGN (deliberately asymmetric)
    * WRITE side  -> firebase-admin SDK + a service-account key.  LOCAL ONLY.
                     The key is secret and must never be committed to git.
    * READ side   -> a single plain HTTPS GET against the RTDB REST API.
                     No SDK and no key needed, so the Streamlit Cloud deploy
                     stays light (it only uses `requests`, already installed).
                     This requires the DB rules to allow public read.
    * If Firebase is not configured, every function is a safe no-op, so the
      whole project still runs fully offline exactly as before.

ONE-TIME SETUP
    1. Create a Firebase project + Realtime Database (see chat / README).
    2. Paste your database URL into DATABASE_URL below (URL is NOT secret).
    3. Download a service-account key -> save as  D:\\final\\firebase_key.json
       (already gitignored). Only the LOCAL backend uses this file.
    4. Set the RTDB security rules to:
           { "rules": { ".read": true, ".write": false } }
    5. Locally:  pip install firebase-admin   (in the `iotfinal` conda env)
"""

import os
import time
import requests

# ── Config ──────────────────────────────────────────────────────────────────
# NOT secret — safe to commit so the cloud dashboard can read it.
# Copy the EXACT url shown in your Realtime Database console (it may end in
# either ".firebaseio.com" or "<region>.firebasedatabase.app").
# You can also override it with the FIREBASE_DB_URL environment variable.
DATABASE_URL = os.environ.get(
    "FIREBASE_DB_URL",
    "https://iot-shield-7ce30-default-rtdb.asia-southeast1.firebasedatabase.app",  # <-- PASTE YOURS HERE
).rstrip("/")

# Path to the service-account key. SECRET. Local writers only. Gitignored.
_CRED_PATH = os.environ.get(
    "FIREBASE_CRED",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "firebase_key.json"),
)

ALERTS_NODE = "live_alerts"
STATS_NODE = "live_stats"
_MIN_PUSH_INTERVAL = 2.0  # seconds — throttle writes to protect the free tier

# ── Internal write-side state ─────────────────────────────────────────────────
_db = None
_init_done = False
_init_ok = False
_last_push = 0.0


def _url_ok():
    return bool(DATABASE_URL) and "YOUR-PROJECT" not in DATABASE_URL


def is_configured():
    """True if a real database URL has been set."""
    return _url_ok()


def _ensure_admin():
    """Lazily initialise firebase-admin for the WRITE side. Returns True on success.
    Imports firebase_admin *inside* the function so the module can still be
    imported on the cloud (where firebase-admin is not installed)."""
    global _init_done, _init_ok, _db
    if _init_done:
        return _init_ok
    _init_done = True

    if not _url_ok():
        print("[firebase] DATABASE_URL not set — sync disabled (writing local files only).")
        return False
    if not os.path.exists(_CRED_PATH):
        print(f"[firebase] Key not found at {_CRED_PATH} — sync disabled (local files only).")
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials, db
        if not firebase_admin._apps:
            cred = credentials.Certificate(_CRED_PATH)
            firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})
        _db = db
        _init_ok = True
        print(f"[firebase] Connected -> {DATABASE_URL}")
    except Exception as e:
        print(f"[firebase] Init failed ({e}) — sync disabled (local files only).")
        _init_ok = False
    return _init_ok


def push(alerts, stats, force=False):
    """WRITE side. Mirror the current alerts list + stats dict to the RTDB.
    Throttled to at most once per _MIN_PUSH_INTERVAL seconds unless force=True
    (use force=True on shutdown / final flush so nothing is lost)."""
    global _last_push
    if not _ensure_admin():
        return False
    now = time.time()
    if not force and (now - _last_push) < _MIN_PUSH_INTERVAL:
        return False
    try:
        _db.reference(ALERTS_NODE).set(alerts)
        _db.reference(STATS_NODE).set(stats)
        _last_push = now
        return True
    except Exception as e:
        print(f"[firebase] Push failed: {e}")
        return False


def fetch(timeout=4):
    """READ side. One HTTPS GET of the whole DB, then pull out the two nodes.
    No SDK / key needed. Returns (alerts_list, stats_dict), or (None, None) if
    Firebase is not configured or unreachable so the caller can fall back to
    the local JSON files."""
    if not _url_ok():
        return None, None
    try:
        r = requests.get(f"{DATABASE_URL}/.json", timeout=timeout)
        r.raise_for_status()
        data = r.json() or {}
        alerts = data.get(ALERTS_NODE) or []
        stats = data.get(STATS_NODE) or {}
        # RTDB can return an array as a dict keyed "0","1",... — normalise it.
        if isinstance(alerts, dict):
            alerts = [alerts[k] for k in sorted(alerts, key=lambda x: int(x))]
        return alerts, stats
    except Exception:
        return None, None

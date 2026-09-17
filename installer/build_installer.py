# build_installer.py
# Builds a one-click Windows installer for IoT Shield Sensor
# Run: python build_installer.py

import os, shutil, subprocess, urllib.request

DIST_DIR    = 'dist'
SENSOR_DIR  = 'sensor_bundle'
NPCAP_URL   = 'https://npcap.com/dist/npcap-1.79.exe'
NPCAP_FILE  = 'npcap_installer.exe'

os.makedirs(SENSOR_DIR, exist_ok=True)

# ── Step 1: Download Npcap installer ──────────────────────────
print("Downloading Npcap installer...")
urllib.request.urlretrieve(NPCAP_URL, 
    os.path.join(SENSOR_DIR, NPCAP_FILE))
print("  Done")

# ── Step 2: Copy model files ───────────────────────────────────
print("Copying model files...")
shutil.copytree('../models', 
    os.path.join(SENSOR_DIR, 'models'),
    dirs_exist_ok=True)
print("  Done")

# ── Step 3: Copy sensor scripts ────────────────────────────────
print("Copying sensor scripts...")
for f in ['step5_live.py', 'firebase_sync.py']:
    shutil.copy(f'../{f}', SENSOR_DIR)
print("  Done")

print("\nAll files ready in sensor_bundle/")
print("Now run PyInstaller on sensor_main.py")
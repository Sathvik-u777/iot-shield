# sensor_main.py
# This becomes the .exe that users download and run.
# It handles admin elevation, a one-time Npcap install, and then runs the
# sensor IN-PROCESS (no subprocess calls back into the frozen exe, and no
# runtime pip install — every dependency is bundled in at build time via
# `pyinstaller --collect-all ...`).

import os
import sys
import subprocess
import ctypes
import winreg
import time


def is_admin():
    """Check if running as administrator"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False


def elevate():
    """Restart self as administrator"""
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas",
        sys.executable, " ".join(sys.argv), None, 1
    )
    sys.exit()


def npcap_installed():
    """Check if Npcap is already installed"""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Npcap"
        )
        winreg.CloseKey(key)
        return True
    except Exception:
        return False


def resource_path(relative_path):
    """
    Absolute path to a bundled resource — works both when run as a normal
    script and when frozen into a PyInstaller onefile exe (where bundled
    --add-data files are extracted to sys._MEIPASS at runtime).
    """
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def install_npcap():
    """
    Install Npcap. The free (non-OEM) Npcap installer does NOT support
    silent install (/S) — it always shows its own setup wizard. This is
    a one-time step: npcap_installed() skips it on every future run.
    """
    print("Installing Npcap driver...")
    print("A one-time Npcap setup window will open now — click through it.")
    print('Make sure "WinPcap API-compatible Mode" stays checked.')
    npcap_path = resource_path("npcap_installer.exe")
    subprocess.run([npcap_path, "/winpcap_mode=yes"], check=True)
    print("  Npcap installed successfully")
    time.sleep(3)


def start_sensor():
    """
    Run the sensor IN-PROCESS. Does NOT shell out to `sys.executable` —
    inside a frozen exe, sys.executable points at this exe itself, not a
    real Python interpreter, so subprocess-based approaches silently break.
    Importing and calling step5_live.main() directly avoids that entirely.
    """
    print("Starting IoT Shield Sensor...")
    print("Dashboard: https://iot-shield.streamlit.app")
    print("Press Ctrl+C to stop\n")

    bundle_dir = resource_path(".")
    sys.path.insert(0, bundle_dir)
    os.chdir(bundle_dir)  # so step5_live's relative models/data paths resolve correctly

    import step5_live
    step5_live.main(reset=True)


if __name__ == "__main__":
    print("=" * 50)
    print("  IoT Shield — Sensor Installer")
    print("  Version 1.0")
    print("=" * 50)
    print()

    # Must run as admin for Npcap install and live packet capture
    if not is_admin():
        print("Requesting administrator privileges...")
        elevate()

    # Install Npcap if not already there (one-time, needs one click-through)
    if npcap_installed():
        print("Npcap already installed — skipping")
    else:
        install_npcap()

    # Run the sensor directly — no external Python process needed
    start_sensor()
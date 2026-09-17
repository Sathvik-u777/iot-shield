# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['sensor_main.py'],
    pathex=[],
    binaries=[],
    datas=[('sensor_bundle/models', 'models'), ('sensor_bundle/npcap_installer.exe', '.'), ('sensor_bundle/step5_live.py', '.'), ('sensor_bundle/firebase_sync.py', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='IoTShield-Sensor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

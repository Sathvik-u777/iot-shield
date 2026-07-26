
"""
step1_prepare.py — IoT Shield (Full Dataset)
Reads 6M row IoT-23 dataset in chunks, maps labels, balances classes,
saves combined.csv ready for training.
Run: python step1_prepare.py
"""
 
import os
import gc
import pandas as pd
 
# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR  = r"D:\final"
DATA_DIR  = os.path.join(BASE_DIR, "data")
INPUT     = os.path.join(DATA_DIR, "dataset.csv")
OUTPUT    = os.path.join(DATA_DIR, "combined.csv")
 
# ─────────────────────────────────────────────
# Label mapping
# ─────────────────────────────────────────────
LABEL_MAP = {
    "Benign":                        "benign",
    "DDoS":                          "ddos",
    "PartOfAHorizontalPortScan":     "portscan",
    "Okiru":                         "malware",
    "Okiru-Attack":                  "malware",
    "Attack":                        "malware",
    "FileDownload":                  "malware",
    "C&C":                           "malware",
    "C&C-HeartBeat":                 "malware",
    "C&C-FileDownload":              "malware",
    "C&C-Torii":                     "malware",
    "C&C-Mirai":                     "malware",
    "C&C-HeartBeat-FileDownload":    "malware",
}
 
# ─────────────────────────────────────────────
# Feature columns
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "ts", "id.orig_p", "id.resp_p", "proto", "service",
    "conn_state", "missed_bytes", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes"
]
 
NUM_COLS = [
    "ts", "id.orig_p", "id.resp_p", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"
]
CAT_COLS = ["proto", "service", "conn_state"]
 
# Samples per class
SAMPLES_PER_CLASS = 150_000
CHUNK_SIZE        = 100_000
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
print("=" * 60)
print("  IoT Shield — Data Preparation (Full Dataset)")
print("=" * 60)
 
print(f"\n[1] Reading {INPUT} in chunks of {CHUNK_SIZE:,} rows ...")
print("    (this avoids loading 6M rows into RAM at once)\n")
 
# Accumulate per-class buckets while reading
class_buckets = {cls: [] for cls in set(LABEL_MAP.values())}
total_read = 0
 
for i, chunk in enumerate(pd.read_csv(INPUT, chunksize=CHUNK_SIZE, low_memory=False)):
    total_read += len(chunk)
 
    # Map labels
    chunk["label"] = chunk["label"].map(LABEL_MAP)
    chunk = chunk.dropna(subset=["label"])
 
    # Keep only needed columns
    keep = [c for c in FEATURE_COLS + ["label"] if c in chunk.columns]
    chunk = chunk[keep]
 
    # Clean numerics
    for col in NUM_COLS:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
 
    # Clean categoricals
    for col in CAT_COLS:
        if col in chunk.columns:
            chunk[col] = chunk[col].fillna("-").astype(str)
 
    # Add to per-class bucket (only keep up to 2x target to save RAM)
    for cls in class_buckets:
        subset = chunk[chunk["label"] == cls]
        if len(subset) > 0:
            class_buckets[cls].append(subset)
 
    print(f"    Chunk {i+1:3d} done — rows read so far: {total_read:>8,}")
    gc.collect()
 
print(f"\n[2] Balancing classes ({SAMPLES_PER_CLASS:,} per class) ...")
balanced_parts = []
 
for cls in sorted(class_buckets.keys()):
    parts = class_buckets[cls]
    if not parts:
        print(f"    {cls:12s}: NO DATA FOUND — skipping")
        continue
    cls_df = pd.concat(parts, ignore_index=True)
    n = min(len(cls_df), SAMPLES_PER_CLASS)
    sampled = cls_df.sample(n=n, random_state=42)
    balanced_parts.append(sampled)
    print(f"    {cls:12s}: {len(cls_df):>10,} available -> {n:>7,} sampled")
    del cls_df
    gc.collect()
 
print(f"\n[3] Combining and shuffling ...")
combined = pd.concat(balanced_parts, ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
 
print(f"    Final dataset: {len(combined):,} rows")
print(f"    Classes: {sorted(combined['label'].unique())}")
 
print(f"\n[4] Saving to {OUTPUT} ...")
combined.to_csv(OUTPUT, index=False)
print(f"    Saved {len(combined):,} rows")
 
print("\n" + "=" * 60)
print("  Done! Run step2_train.py next.")
print("=" * 60)
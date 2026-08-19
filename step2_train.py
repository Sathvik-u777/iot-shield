"""
step2_train.py — IoT Shield (Full Dataset, leakage-fixed)
Trains Random Forest + XGBoost ensemble on 600k balanced rows.

Fixes applied vs previous version:
  1. 'ts' (raw timestamp) is dropped as a MODEL FEATURE — it was letting the
     model learn "which time block" instead of real traffic behavior.
     'ts' is still used, but only to sort rows for a chronological split.
  2. Train/test split is now CHRONOLOGICAL (earlier flows = train, later
     flows = test) instead of a random row split. A random split let
     near-duplicate flows from the same attack session leak between
     train and test.
  3. Exact duplicate rows are dropped before splitting.
  4. Removed the duplicated real-world-distribution block (was run twice).

Run: python step2_train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = r"D:\final"
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT      = os.path.join(DATA_DIR, "combined.csv")

# ─────────────────────────────────────────────
# Feature columns
#   NOTE: 'ts' is intentionally EXCLUDED from MODEL_FEATURE_COLS.
#   It's only used to sort rows for a chronological train/test split.
# ─────────────────────────────────────────────
ALL_COLS = [
    "ts", "id.orig_p", "id.resp_p", "proto", "service",
    "conn_state", "missed_bytes", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes"
]
MODEL_FEATURE_COLS = [
    "id.orig_p", "id.resp_p", "proto", "service",
    "conn_state", "missed_bytes", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes"
]
CAT_COLS = ["proto", "service", "conn_state"]

print("=" * 60)
print("  IoT Shield — Model Training (Full Dataset, leakage-fixed)")
print("=" * 60)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print(f"\n[1] Loading {INPUT} ...")
df = pd.read_csv(INPUT)
print(f"    Rows: {len(df):,}  |  Columns: {len(df.columns)}")
print(f"    Classes: {sorted(df['label'].unique())}")

# ─────────────────────────────────────────────
# Drop exact duplicate rows (same feature values + label)
# ─────────────────────────────────────────────
print("\n[2] Checking for duplicate rows ...")
before = len(df)
df = df.drop_duplicates(subset=MODEL_FEATURE_COLS + ["label"]).reset_index(drop=True)
after = len(df)
print(f"    Removed {before - after:,} duplicate rows ({before:,} -> {after:,})")

# ─────────────────────────────────────────────
# Sort chronologically BEFORE splitting
# ─────────────────────────────────────────────
print("\n[3] Sorting rows chronologically by 'ts' ...")
df = df.sort_values("ts").reset_index(drop=True)

# ─────────────────────────────────────────────
# Encode categoricals
# ─────────────────────────────────────────────
print("\n[4] Encoding categorical columns ...")
cat_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    cat_encoders[col] = le
    print(f"    {col}: {len(le.classes_)} unique values")

joblib.dump(cat_encoders, os.path.join(MODEL_DIR, "cat_encoders.pkl"))
print("    Saved cat_encoders.pkl")

# ─────────────────────────────────────────────
# Encode labels
# ─────────────────────────────────────────────
print("\n[5] Encoding labels ...")
label_enc = LabelEncoder()
df["label_enc"] = label_enc.fit_transform(df["label"])
print(f"    Classes: {list(label_enc.classes_)}")
joblib.dump(label_enc, os.path.join(MODEL_DIR, "label_enc.pkl"))
print("    Saved label_enc.pkl")

# ─────────────────────────────────────────────
# Chronological train/test split (80/20), done PER CLASS.
#   Each class in IoT-23 is captured in its own distinct time block
#   (e.g. all malware flows happen earlier than all portscan flows),
#   so a single global chronological cut drops classes entirely from
#   one side of the split. Splitting each class's own timeline 80/20
#   keeps all 4 classes in both sets while still separating train from
#   test in time within each class (no same-session flow appears in
#   both).
# ─────────────────────────────────────────────
print("\n[6] Splitting data chronologically per class (80/20) ...")
train_parts, test_parts_split = [], []
for cls in sorted(df["label"].unique()):
    cls_df = df[df["label"] == cls].sort_values("ts")
    cut = int(len(cls_df) * 0.8)
    train_parts.append(cls_df.iloc[:cut])
    test_parts_split.append(cls_df.iloc[cut:])

train_df = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_parts_split).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"    Train: {len(train_df):,}  |  Test: {len(test_df):,}")
print("    Train class distribution:")
print(train_df["label"].value_counts())
print("    Test class distribution:")
print(test_df["label"].value_counts())

X_train_raw = train_df[MODEL_FEATURE_COLS].values
y_train = train_df["label_enc"].values
X_test_raw = test_df[MODEL_FEATURE_COLS].values
y_test = test_df["label_enc"].values

joblib.dump(MODEL_FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))
print(f"\n    Feature columns saved: {MODEL_FEATURE_COLS}")

# ─────────────────────────────────────────────
# Scale (fit ONLY on train, transform both)
# ─────────────────────────────────────────────
print("\n[7] Scaling features (fit on train only) ...")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("    Saved scaler.pkl")

# ─────────────────────────────────────────────
# Rebalance test set to real-world IoT traffic distribution
# Real IoT networks: mostly benign with occasional attacks
# ─────────────────────────────────────────────
print("\n[8] Applying real-world test distribution ...")
test_eval_df = pd.DataFrame(X_test, columns=MODEL_FEATURE_COLS)
test_eval_df["label"] = y_test

real_world = {"benign": 0.60, "ddos": 0.15, "malware": 0.10, "portscan": 0.15}
test_parts = []
total_test = min(20000, len(test_eval_df))

for cls_name, ratio in real_world.items():
    cls_idx = label_enc.transform([cls_name])[0]
    cls_rows = test_eval_df[test_eval_df["label"] == cls_idx]
    n = int(total_test * ratio)
    n = min(n, len(cls_rows))
    sampled = cls_rows.sample(n=n, random_state=42)
    test_parts.append(sampled)
    print(f"    {cls_name:12s}: {n:,} samples ({ratio*100:.0f}%)")

test_resampled = pd.concat(test_parts).sample(frac=1, random_state=42)
X_test = test_resampled.drop("label", axis=1).values
y_test = test_resampled["label"].values
print(f"    Total test set: {len(X_test):,} rows")

# ─────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────
print("\n[9] Training Random Forest ...")
print("    (this will take 5-10 minutes on your laptop)")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"    RF Accuracy: {rf_acc*100:.2f}%")
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
print("    Saved rf_model.pkl")

# ─────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────
print("\n[10] Training XGBoost ...")
print("    (this will take 5-10 minutes on your laptop)")
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
    verbosity=0
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print(f"    XGBoost Accuracy: {xgb_acc*100:.2f}%")
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))
print("    Saved xgb_model.pkl")

# ─────────────────────────────────────────────
# Ensemble evaluation
# ─────────────────────────────────────────────
print("\n[11] Ensemble evaluation (RF x0.45 + XGB x0.55) ...")
rf_proba  = rf.predict_proba(X_test)
xgb_proba = xgb.predict_proba(X_test)
ens_proba = 0.45 * rf_proba + 0.55 * xgb_proba
ens_preds = np.argmax(ens_proba, axis=1)
ens_acc   = accuracy_score(y_test, ens_preds)
print(f"    Ensemble Accuracy: {ens_acc*100:.2f}%")

# ─────────────────────────────────────────────
# Full report
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  Random Forest accuracy : {rf_acc*100:.2f}%")
print(f"  XGBoost accuracy       : {xgb_acc*100:.2f}%")
print(f"  Ensemble accuracy      : {ens_acc*100:.2f}%")

print("\n  Classification Report (Ensemble):")
print(classification_report(
    y_test, ens_preds,
    target_names=label_enc.classes_
))

print("  Confusion Matrix (Ensemble):")
cm = confusion_matrix(y_test, ens_preds)
print(f"  Classes: {list(label_enc.classes_)}")
print(cm)

print("\n" + "=" * 60)
print("  All models saved to D:\\final\\models\\")
print("  Done! Run step3_ai.py next.")
print("=" * 60)
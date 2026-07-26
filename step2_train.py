"""
step2_train.py — IoT Shield (Full Dataset)
Trains Random Forest + XGBoost ensemble on 600k balanced rows.
Run: python step2_train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "ts", "id.orig_p", "id.resp_p", "proto", "service",
    "conn_state", "missed_bytes", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes"
]
CAT_COLS = ["proto", "service", "conn_state"]

print("=" * 60)
print("  IoT Shield — Model Training (Full Dataset)")
print("=" * 60)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print(f"\n[1] Loading {INPUT} ...")
df = pd.read_csv(INPUT)
print(f"    Rows: {len(df):,}  |  Columns: {len(df.columns)}")
print(f"    Classes: {sorted(df['label'].unique())}")

# ─────────────────────────────────────────────
# Encode categoricals
# ─────────────────────────────────────────────
print("\n[2] Encoding categorical columns ...")
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
print("\n[3] Encoding labels ...")
label_enc = LabelEncoder()
df["label_enc"] = label_enc.fit_transform(df["label"])
print(f"    Classes: {list(label_enc.classes_)}")
joblib.dump(label_enc, os.path.join(MODEL_DIR, "label_enc.pkl"))
print("    Saved label_enc.pkl")

# ─────────────────────────────────────────────
# Features and labels
# ─────────────────────────────────────────────
X = df[FEATURE_COLS].values
y = df["label_enc"].values

# Save feature cols
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))
print(f"\n    Feature columns saved: {FEATURE_COLS}")

# ─────────────────────────────────────────────
# Scale
# ─────────────────────────────────────────────
print("\n[4] Scaling features ...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("    Saved scaler.pkl")

# ─────────────────────────────────────────────
# Train / test split
# ─────────────────────────────────────────────
print("\n[5] Splitting data (80/20) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────
print("\n[6] Training Random Forest ...")
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
print("\n[7] Training XGBoost ...")
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
print("\n[8] Ensemble evaluation (RF x0.45 + XGB x0.55) ...")
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
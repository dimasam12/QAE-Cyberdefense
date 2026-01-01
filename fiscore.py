# =========================================================
# QAE FINAL EVALUATION SCRIPT
# Accuracy, Precision, Recall, F1-score
# =========================================================

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("QAE FINAL EVALUATION")
print("=" * 60)

# =========================================================
# 1. FILE CONFIG
# =========================================================
QAE_RESULT_FILE = "ae_thresholded_results.csv"
LABEL_FILE = "test_ddos_normal.csv"

# =========================================================
# 2. LOAD FILES
# =========================================================
df_qae = pd.read_csv(QAE_RESULT_FILE)
df_label = pd.read_csv(LABEL_FILE, low_memory=False)

print(f"✓ QAE result loaded   : {df_qae.shape}")
print(f"✓ Label data loaded  : {df_label.shape}")

# =========================================================
# 3. ALIGN DATA LENGTH
# =========================================================
min_len = min(len(df_qae), len(df_label))

df_qae = df_qae.iloc[:min_len].reset_index(drop=True)
df_label = df_label.iloc[:min_len].reset_index(drop=True)

print(f"✓ Data aligned to {min_len} samples")

# =========================================================
# 4. BUILD GROUND TRUTH LABEL
# attack_cat:
#   NaN  -> Normal (0)
#   NOT NaN -> Attack (1)
# =========================================================
y_true = df_label["attack_cat"].notna().astype(int)
y_pred = df_qae["predicted_label"].astype(int)

# =========================================================
# 5. METRICS
# =========================================================
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)
cm        = confusion_matrix(y_true, y_pred)

# =========================================================
# 6. REPORT
# =========================================================
print("\n================ EVALUATION RESULT ================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Attack"],
    zero_division=0
))

# =========================================================
# 7. SAVE EVALUATION RESULT
# =========================================================
eval_summary = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "TN": cm[0, 0],
    "FP": cm[0, 1],
    "FN": cm[1, 0],
    "TP": cm[1, 1],
}

df_eval = pd.DataFrame([eval_summary])
df_eval.to_csv("qae_evaluation_metrics.csv", index=False)

print("\n✓ Evaluation metrics saved to qae_evaluation_metrics.csv")
print("=" * 60)

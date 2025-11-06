import numpy as np
import json
from sklearn.metrics import confusion_matrix, roc_auc_score

# Load your stored raw predictions
data = np.load("results_multi/test_raw_arrays.npz")
y_true = data["y_true"]
y_prob = data["y_prob"]

# Try multiple thresholds
thresholds = np.linspace(0.1, 0.9, 9)
print("Threshold | Sensitivity | Specificity | Accuracy")
print("---------------------------------------------")

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    accuracy = (tp + tn) / len(y_true)
    print(f"{t:9.2f} | {sensitivity:11.2f} | {specificity:11.2f} | {accuracy:8.2f}")


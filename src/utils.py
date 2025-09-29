from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def aggregate_subject_probs(subject_ids, probs):
    """
    Aggregerer slice-probabiliteter per subject ved gjennomsnitt.
    Returnerer:
      subj_ids_unique, agg_probs
    """
    by_subj = defaultdict(list)
    for sid, p in zip(subject_ids, probs):
        by_subj[sid].append(p)
    subj_ids = []
    agg = []
    for sid, plist in by_subj.items():
        subj_ids.append(sid)
        agg.append(np.mean(plist, axis=0))
    return subj_ids, np.vstack(agg)

def compute_metrics(y_true, y_prob):
    """
    y_true: (N,) int labels {0,1}
    y_prob: (N,2) probabilities for two classes
    """
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob[:,1])
    except Exception:
        auc = float('nan')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn + 1e-9)  # recall for class 1
    spec = tn / (tn + fp + 1e-9)  # recall for class 0
    return {"accuracy": acc, "auc": auc, "sensitivity": sens, "specificity": spec}

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.dataset import SliceDataset
from src.model import ResNet18MRI  
from src.utils import aggregate_subject_probs, compute_metrics

from sklearn.metrics import roc_curve, auc, confusion_matrix


def _make_loader(csv_path, multi_slice, batch_size=32, num_workers=2, shuffle=False):
    ds = SliceDataset(csv_path, multi_slice=multi_slice)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=False)
    return ds, loader


#collect model outputs
@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    all_probs, all_labels, all_subj = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].numpy()
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y)
        all_subj.extend(batch["subject_id"])
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_labels, all_probs, np.array(all_subj)


#plotting function
def plot_roc(y_true, y_score, title, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, color="darkorange", label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return roc_auc


def plot_confmat(y_true, y_pred, title, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Control(0)", "Demented(1)"])
    plt.yticks(tick_marks, ["Control(0)", "Demented(1)"])
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                sensitivity=sensitivity, specificity=specificity)


# evaluation split
def evaluate_split(model, csv_path, device, multi_slice, tag, outdir):
    os.makedirs(outdir, exist_ok=True)
    ds, loader = _make_loader(csv_path, multi_slice=multi_slice, batch_size=64, num_workers=2, shuffle=False)
    y_true_slice, probs_slice, subj_ids = collect_probs(model, loader, device)

    metrics_slice = compute_metrics(y_true_slice, probs_slice)

    # subject-level aggregation
    subj_list, probs_subject = aggregate_subject_probs(subj_ids.tolist(), probs_slice)
    import pandas as pd
    y_true_subject = (
        pd.DataFrame({"subject_id": subj_ids, "label": y_true_slice})
        .groupby("subject_id")["label"]
        .first()
        .reindex(subj_list)
        .values
    )
    metrics_subject = compute_metrics(y_true_subject, probs_subject)

    # ROC
    roc_slice = plot_roc(y_true_slice, probs_slice[:, 1],
                         f"{tag} ROC (slice)", os.path.join(outdir, f"{tag}_roc_slice.png"))
    roc_subject = plot_roc(y_true_subject, probs_subject[:, 1],
                           f"{tag} ROC (subject)", os.path.join(outdir, f"{tag}_roc_subject.png"))

    # confusion matrices
    y_pred_slice = (probs_slice[:, 1] >= 0.5).astype(int)
    y_pred_subject = (probs_subject[:, 1] >= 0.5).astype(int)
    cm_slice = plot_confmat(y_true_slice, y_pred_slice,
                            f"{tag} Confusion Matrix (slice)", os.path.join(outdir, f"{tag}_cm_slice.png"))
    cm_subject = plot_confmat(y_true_subject, y_pred_subject,
                              f"{tag} Confusion Matrix (subject)", os.path.join(outdir, f"{tag}_cm_subject.png"))

    #save results
    np.savez(os.path.join(outdir, f"{tag}_raw_arrays.npz"),
             y_true_slice=y_true_slice, probs_slice=probs_slice, subj_ids=subj_ids,
             y_true_subject=y_true_subject, probs_subject=probs_subject)

    summary = {
        "slice": {**metrics_slice, "roc_auc": roc_slice, **cm_slice},
        "subject": {**metrics_subject, "roc_auc": roc_subject, **cm_subject}
    }

    with open(os.path.join(outdir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{tag}] AUC (slice)={roc_slice:.3f} | AUC (subject)={roc_subject:.3f}")
    print(f"[{tag}] Saved plots + summary to {outdir}")
    return summary


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/best_model.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--multi_slice", action="store_true")
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--val_csv", default="data/val.csv")
    ap.add_argument("--test_csv", default="data/test.csv")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    in_channels = 3 if args.multi_slice else 1
    model = ResNet18MRI(in_channels=in_channels, num_classes=2, pretrained=False).to(device)  # ✅ new model
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    outdir = "results"
    os.makedirs(outdir, exist_ok=True)

    _ = evaluate_split(model, args.val_csv, device, args.multi_slice, "val", outdir)
    _ = evaluate_split(model, args.test_csv, device, args.multi_slice, "test", outdir)


if __name__ == "__main__":
    main()

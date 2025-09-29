import json
from typing import Dict
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd

from .dataset import SliceDataset
from .model import SimpleCNN
from .utils import aggregate_subject_probs, compute_metrics

def _make_loader(csv_path, multi_slice, batch_size, num_workers, shuffle):
    ds = SliceDataset(csv_path, multi_slice=multi_slice)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=False)
    return ds, loader

@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    model.eval()
    all_probs = []
    all_labels = []
    all_subj = []
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

    # slice-level metrics
    metrics_slice = compute_metrics(all_labels, all_probs)

    # subject-level metrics
    subj_ids, agg_probs = aggregate_subject_probs(all_subj, all_probs)
    # For subject-level ground truth, ta første forekomst av label per subject (alle slices har samme label)
    subj_df = pd.DataFrame({"subject_id": all_subj, "label": all_labels})
    subj_true = subj_df.groupby("subject_id")["label"].first().reindex(subj_ids).values
    metrics_subject = compute_metrics(subj_true, agg_probs)

    return {
        "slice": metrics_slice,
        "subject": metrics_subject
    }

def train_and_evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print(f"Bruker device: {device}")

    # Data
    train_ds, train_loader = _make_loader(args.train_csv, args.multi_slice, args.batch_size, args.num_workers, shuffle=True)
    val_ds, val_loader = _make_loader(args.val_csv, args.multi_slice, args.batch_size, args.num_workers, shuffle=False)
    test_ds, test_loader = _make_loader(args.test_csv, args.multi_slice, args.batch_size, args.num_workers, shuffle=False)

    in_channels = 3 if args.multi_slice else 1
    model = SimpleCNN(in_channels=in_channels, num_classes=2).to(device)

    # Optimizer/loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_path = "results/best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for batch in pbar:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)

        # Val
        val_metrics = evaluate(model, val_loader, device)
        val_auc = val_metrics["subject"]["auc"]  # bruk subject-level AUC for model selection
        print(f"[Val] Loss={epoch_loss:.4f} | Slice Acc={val_metrics['slice']['accuracy']:.3f}, AUC={val_metrics['slice']['auc']:.3f} | Subject Acc={val_metrics['subject']['accuracy']:.3f}, AUC={val_auc:.3f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"state_dict": model.state_dict(),
                        "in_channels": in_channels}, best_path)
            print(f"🧠 Ny beste modell lagret til {best_path} (Val subject AUC={best_val_auc:.3f})")

    # Evaluer på test med beste modell
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"[Test] Slice Acc={test_metrics['slice']['accuracy']:.3f}, AUC={test_metrics['slice']['auc']:.3f} | Subject Acc={test_metrics['subject']['accuracy']:.3f}, AUC={test_metrics['subject']['auc']:.3f}")

    # lagre metrics
    out = {"val": val_metrics, "test": test_metrics, "best_val_subject_auc": best_val_auc}
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("📄 Metrics lagret i results/metrics.json")

import os
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
from .model import ResNet18MRI
from .utils import aggregate_subject_probs, compute_metrics

def _make_loader(csv_path, multi_slice, batch_size, num_workers, shuffle,train=False):
    ds = SliceDataset(csv_path, multi_slice=multi_slice,train=train)
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
    #train_ds, train_loader = _make_loader(args.train_csv, args.multi_slice, args.batch_size, args.num_workers, shuffle=True)
    from torch.utils.data import WeightedRandomSampler
    import numpy as np

    train_ds, _ = _make_loader(args.train_csv, args.multi_slice, args.batch_size, args.num_workers, shuffle=False)
    train_ds.train = True  # Enable data augmentation for training

    # compute class weights
    labels = np.array([int(l) for l in train_ds.df["label"]])
    class_sample_counts = np.bincount(labels)
    weights = 1.0 / class_sample_counts
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    val_ds, val_loader = _make_loader(args.val_csv, args.multi_slice, args.batch_size, args.num_workers, shuffle=False)
    test_ds, test_loader = _make_loader(args.test_csv, args.multi_slice, args.batch_size, args.num_workers, shuffle=False)

    in_channels = 3 if args.multi_slice else 1
    model = ResNet18MRI(in_channels=in_channels, num_classes=2, pretrained=True).to(device)


    # Optimizer/loss
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Compute class weights based balance
    class_counts = np.bincount(labels)
    pos_boost = 1.5  # makes Demented (class 1) more important
    w = 1.0 / class_counts.astype(np.float32)
    w[1] *= pos_boost  # boost Demented weight
    class_weights = torch.tensor([0.8, 1.2], dtype=torch.float32).to(device)
    # class_weights = torch.tensor(w / w.sum() * 2.0, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    #criterion = nn.CrossEntropyLoss()
    # Compute class weights based on dataset balance
    #class_counts = np.bincount(labels)
    # class weights for loss (gir mer vekt til AD-klassen)
    #class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32).to(device)
    #class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32).to(device)
    #class_weights = torch.tensor([1.0, 2.2], dtype=torch.float32).to(device)
    #class_weights = torch.tensor([1.0, class_counts[0] / max(1, class_counts[1])], dtype=torch.float32).to(device)
    #criterion = nn.CrossEntropyLoss(weight=class_weights)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",factor=0.5,patience=3,min_lr=1e-6)
    best_val_auc = -1.0
    best_path = "results/best_model.pt"
    epochs_no_improve=0
    patience_es=10
    val_auc = 0.0  # init to avoid UnboundLocalError
    

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for step, batch in enumerate(pbar):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            
            if step %100==0:
                avg_loss=running_loss/((step+1)*train_loader.batch_size)
                print(f"Epoch{epoch},Step{step}, Loss={loss.item():.4f}, AvgLoss={avg_loss:.4f}")
            
            pbar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)

        # Val
        val_metrics = evaluate(model, val_loader, device)
        val_auc = val_metrics["subject"]["auc"]  # bruk subject-level AUC for model selection
        print(f"[Val] Loss={epoch_loss:.4f} | Slice Acc={val_metrics['slice']['accuracy']:.3f}, AUC={val_metrics['slice']['auc']:.3f} | Subject Acc={val_metrics['subject']['accuracy']:.3f}, AUC={val_auc:.3f}")

        #scheduler + early stopping
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve=0
            torch.save({"state_dict": model.state_dict(),
                        "in_channels": in_channels}, best_path)
            print(f"Ny beste modell lagret til {best_path} (Val subject AUC={best_val_auc:.3f})")
        else:
            epochs_no_improve+=1
            print(f"no improvement in val AUC for {epochs_no_improve} epochs")

            if epochs_no_improve>= patience_es:
                print("early stopping triggered (no val AUC improvment)")
                break
        
        scheduler.step(val_auc)

    # evaluate best model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"[Test] Slice Acc={test_metrics['slice']['accuracy']:.3f}, AUC={test_metrics['slice']['auc']:.3f} | Subject Acc={test_metrics['subject']['accuracy']:.3f}, AUC={test_metrics['subject']['auc']:.3f}")

    # metrices
    out = {"val": val_metrics, "test": test_metrics, "best_val_subject_auc": best_val_auc}
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(" Metrics lagret i results/metrics.json")

import torch
from torch.utils.data import DataLoader
from src.dataset import SliceDataset

# ✅ use the same parameters you plan for training
csv_path = "data/train.csv"
dataset = SliceDataset(csv_path, multi_slice=True)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

batch = next(iter(loader))
images = batch["image"]
labels = batch["label"]

print(f"🧠 Batch image tensor shape: {images.shape}")
print(f"🏷️  Batch labels: {labels.tolist()}")
print(f"Unique labels in batch: {labels.unique().tolist()}")
print(f"Number of distinct subjects in batch: {len(set(batch['subject_id']))}")

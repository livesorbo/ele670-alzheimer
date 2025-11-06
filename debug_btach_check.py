import torch
from torch.utils.data import DataLoader
from src.dataset import SliceDataset

csv_path = "data/train.csv"
dataset = SliceDataset(csv_path, multi_slice=True)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

batch = next(iter(loader))
images = batch["image"]
labels = batch["label"]


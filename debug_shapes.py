from torch.utils.data import DataLoader
from src.dataset import SliceDataset

# point to your train.csv
csv_path = "data/train.csv"

# try both single-slice and multi-slice
ds = SliceDataset(csv_path, multi_slice=True)
loader = DataLoader(ds, batch_size=4, shuffle=False)

batch = next(iter(loader))
x, y, subj = batch["image"], batch["label"], batch["subject_id"]

print("Batch image shape:", x.shape)   # expect (B, C, H, W)
print("Batch label shape:", y.shape)   # expect (B,)
print("First subject ids:", subj[:4])


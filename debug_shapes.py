from torch.utils.data import DataLoader
from src.dataset import SliceDataset

csv_path = "data/train.csv"
ds = SliceDataset(csv_path, multi_slice=True)
loader = DataLoader(ds, batch_size=4, shuffle=False)

batch = next(iter(loader))
x, y, subj = batch["image"], batch["label"], batch["subject_id"]

print("Batch image shape:", x.shape)   
print("Batch label shape:", y.shape) 
print("First subject ids:", subj[:4])


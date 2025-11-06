import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import random

#used to check the data

# read
df = pd.read_csv("data/train.csv")


sample = df.sample(1).iloc[0]
nifti_path = sample["nifti_path"]
subject_id = sample["subject_id"]

print(f" Tester fil: {nifti_path} (Subject: {subject_id})")

# open NIfTI
img = nib.load(nifti_path)
data = img.get_fdata()
print(" Shape på volumet orginalt:", data.shape)

#right dimension?
if data.ndim == 4 and data.shape[-1] == 1:
    data = np.squeeze(data, axis=-1)
    print("Shape etter squeeze:", data.shape)

slices = [
    data[data.shape[0] // 2, :, :],  
    data[:, data.shape[1] // 2, :],  
    data[:, :, data.shape[2] // 2]   
]

# plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ["Sagittal", "Coronal", "Axial"]

for i, sl in enumerate(slices):
    axes[i].imshow(sl.T, cmap="gray", origin="lower")
    axes[i].set_title(titles[i])
    axes[i].axis("off")

plt.suptitle(f"Subject: {subject_id}", fontsize=14)
plt.show()

import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import random

# Les treningssettet
df = pd.read_csv("data/train.csv")

# Velg et tilfeldig volum
sample = df.sample(1).iloc[0]
nifti_path = sample["nifti_path"]
subject_id = sample["subject_id"]

print(f" Tester fil: {nifti_path} (Subject: {subject_id})")

# Åpne NIfTI
img = nib.load(nifti_path)
data = img.get_fdata()
print(" Shape på volumet orginalt:", data.shape)

# Fjern ekstra dimensjon hvis den finnes
if data.ndim == 4 and data.shape[-1] == 1:
    data = np.squeeze(data, axis=-1)
    print("Shape etter squeeze:", data.shape)

# Finn midtsnitt
slices = [
    data[data.shape[0] // 2, :, :],  
    data[:, data.shape[1] // 2, :],  
    data[:, :, data.shape[2] // 2]   
]

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ["Sagittal", "Coronal", "Axial"]

for i, sl in enumerate(slices):
    axes[i].imshow(sl.T, cmap="gray", origin="lower")
    axes[i].set_title(titles[i])
    axes[i].axis("off")

plt.suptitle(f"Subject: {subject_id}", fontsize=14)
plt.show()

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = "data"

# Finn alle .nii.gz filer
files = glob.glob(os.path.join(data_dir, "*.nii.gz"))
print(f"Fant {len(files)} NIfTI-filer")

rows = []
for f in files:
    subject_id = os.path.basename(f).split(".")[0]
    label = 0  #dummy-label
    rows.append([subject_id, f, label])

df = pd.DataFrame(rows, columns=["subject_id", "nifti_path", "label"])

# Split train/val/test
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)



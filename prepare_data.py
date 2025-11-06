import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# paths 
data_dir = "data"
excel_path = os.path.join(data_dir, "oasis_cross-sectional.xlsx")

# metadata
meta = pd.read_excel(excel_path)
print(f"Loaded metadata with {len(meta)} subjects")

# using the CDR to determine AD or HC
meta["label"] = (meta["CDR"] > 0).astype(int)

# find available MRI files
files = glob.glob(os.path.join(data_dir, "*.nii.gz"))
print(f"Found {len(files)} NIfTI files")

rows = []
missing = 0
matched = 0

for f in files:
    subject_id = os.path.basename(f).split("_")[1]
    subj_name = f"OAS1_{subject_id}"

    match = meta[meta["ID"].astype(str).str.contains(subject_id)]
    if not match.empty:
        label = int(match["label"].values[0])
        rows.append([subj_name, f, label])
        matched += 1
    else:
        missing += 1

print(f"Matched {matched} MRI files with metadata, missing {missing}")

# dataframe
df = pd.DataFrame(rows, columns=["subject_id", "nifti_path", "label"])

# train/val/test
train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

# save CSV
train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
val.to_csv(os.path.join(data_dir, "val.csv"), index=False)
test.to_csv(os.path.join(data_dir, "test.csv"), index=False)

#print("\n Saved splits:")
#print(f"  Train: {len(train)} samples")
#print(f"  Val:   {len(val)} samples")
#print(f"  Test:  {len(test)} samples")

#print("\nLabel counts (train):")
#print(train['label'].value_counts())

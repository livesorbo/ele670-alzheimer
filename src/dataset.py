import os
from typing import List, Tuple, Dict
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset

def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = x.mean()
    s = x.std()
    return (x - m) / (s + eps) if s > 0 else x - m

class SliceDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 multi_slice: bool = False,
                 axis: str = "z",
                 central_fraction: float = 0.6,
                 train: bool=False):
        super().__init__()
        self.train=train
        assert axis in {"x", "y", "z"}
        self.df = pd.read_csv(csv_path)
        if not {"subject_id", "nifti_path", "label"}.issubset(self.df.columns):
            raise ValueError("CSV må ha kolonnene: subject_id,nifti_path,label")
        self.multi_slice = multi_slice
        self.axis = axis
        self.central_fraction = central_fraction
        
        self.index: List[Tuple[int, int]] = []
        for i, row in self.df.iterrows():
            path = row["nifti_path"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Finner ikke fil: {path}")
            img = nib.load(path)
            shape = img.shape
            axis_dim = {"x": 0, "y": 1, "z": 2}[axis]
            n = shape[axis_dim]
            
            start = int((1 - self.central_fraction) / 2 * n)
            end = int(n - start)
            for s in range(start, end):
                self.index.append((i, s))
        
        self._cache: Dict[int, np.ndarray] = {}

    def __len__(self):
        return len(self.index)

    def _load_volume(self, row_idx: int) -> np.ndarray:
        if row_idx in self._cache:
            return self._cache[row_idx]
        path = self.df.iloc[row_idx]["nifti_path"]
        data = nib.load(path).get_fdata().astype(np.float32)
        
        if data.ndim==4 and data.shape[-1]==1:
            data=data[...,0]
        data = _zscore(data)
        self._cache[row_idx] = data
        return data

    def _get_slice(self, vol: np.ndarray, s: int, axis: str) -> np.ndarray:
        if axis == "z":
            sl = vol[:, :, s]
        elif axis == "y":
            sl = vol[:, s, :]
        else:  # x
            sl = vol[s, :, :]
        
        sl = _zscore(sl)
        return sl

    def __getitem__(self, idx: int):
        row_idx, s = self.index[idx]
        row = self.df.iloc[row_idx]
        vol = self._load_volume(row_idx)
        C = 3 if self.multi_slice else 1

        if self.multi_slice:
        
            s_idxs = [max(0, s - 1), s, min(vol.shape[{"x": 0, "y": 1, "z": 2}[self.axis]] - 1, s + 1)]
            slices = []
            for si in s_idxs:
                slices.append(self._get_slice(vol, si, self.axis))
            arr = np.stack(slices, axis=0)  # (3, H, W)
        else:
            sl = self._get_slice(vol, s, self.axis)
            arr = sl[None, ...]  # (1, H, W)

        tensor = torch.from_numpy(arr.astype(np.float32))

    #data aug. 
        if self.train:
            import random
            import torch.nn.functional as F

        # Random horizontal flip
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])

        # Random vertical flip
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[1])

        # Random scaling (zoom)
            if random.random() < 0.3:
                scale_factor = random.uniform(0.9, 1.1)
                tensor = F.interpolate(tensor.unsqueeze(0), scale_factor=scale_factor,
                                    mode='bilinear', align_corners=False).squeeze(0)
                tensor = F.interpolate(tensor.unsqueeze(0), size=(176, 208),
                                    mode='bilinear', align_corners=False).squeeze(0)

        # Gaussian noise
            if random.random() < 0.3:
                noise = torch.randn_like(tensor) * 0.02
                tensor = torch.clamp(tensor + noise, 0, 1)


        label = int(row["label"])
        subj = str(row["subject_id"])
        return {"image": tensor, "label": label, "subject_id": subj}

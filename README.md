####
# Alzheimer’s Disease Classification from MRI (ELE670 Project)

## Project Description
This project aims to classify patients with Alzheimer’s Disease using MRI scans from the OASIS-1 dataset. The approach uses 2D CNNs with multi-slice inputs and subject-level aggregation.

## Progress
- Project structure and GitHub repository created  
- OASIS-1 dataset converted from `.hdr/.img` to NIfTI (`.nii.gz`)  
- Data split into train (70%), val (15%), and test (15%) with CSV files  
- Data quality verified with slice visualization  
- SSH server and GPU environment configured (PyTorch, MONAI, NiBabel)  

## Next Steps
- Implement baseline 2D CNN on slices  
- Add subject-level aggregation (average / majority vote)  
- Extend to multi-slice input  
- Apply data augmentation  
- Experiment with transfer learning  
- Train and evaluate models, generate results  

## Environment Setup
```bash
conda create -n ELE670 python=3.10
conda activate ELE670
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel scikit-learn matplotlib pandas






#GIT: https://github.com/livesorbo/ele670-alzheimer.git 
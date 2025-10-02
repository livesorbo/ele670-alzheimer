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

## forward:
* Implement baseline 2D CNN  
* Add subject-level aggregation
* Extend to multi-slice input  
* Apply augmentation  
* Train and evaluate models, generate results  

## Environment Setup
```bash
uenv verbose cuda-11.8.0
uenv verbose cudnn-11.x-8.8.0 
uenv verbose miniconda3-py312 
conda create --name ELE670
conda activate ELE670
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy
pip3 install matplotlib
pip3 install pandas





#GIT: https://github.com/livesorbo/ele670-alzheimer.git 
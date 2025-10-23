####
# Alzheimer’s Disease Classification from MRI (ELE670 Project)

## Project Description
This project aims to classify patients with Alzheimer’s Disease using MRI scans from the OASIS-1 dataset. The approach uses 2D CNNs with multi-slice inputs, data augmentation, and subject level aggregation for improved accuracy and robustness. The final model uses a ResNet-18 backbone with transfer learning and is using clinical metrices as AUC, sensitivity, and specificity 

## Progress
- Project structure and GitHub repository created  
- OASIS-1 dataset converted from `.hdr/.img` to NIfTI (`.nii.gz`)  
- Data split into train (70%), val (15%), and test (15%) with CSV files  
- Data quality verified with slice visualization  
- SSH server and GPU environment configured (PyTorch, MONAI, NiBabel)  
- Implemented baseline 2D CNN
- Added multi-slice input
- Implemented subject-level aggregation
- Added data augmentation(flips, scaling and gaussian noise)
- Integrated ResNet-18(transfer learning)
- Training and evaluation ongoing

## forward:
- Generate results, matrices and curves, docoment and discuss final results

## Environment Setup
This project was developed and trained on a GPU-enables SSH server.
Below is the environment setup used:
```bash
#Loading necessary modules
uenv verbose cuda-11.8.0
uenv verbose cudnn-11.x-8.8.0 
uenv verbose miniconda3-py312 
#Creating and activating environment
conda create --name ELE670
conda activate ELE670

#Install
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy
pip3 install matplotlib
pip3 install pandas

#Training:
export CUDA_VISIBLE_DEVICES=2
nohup python3 main.py --epochs 10 --batch_size 16 --lr 1e-4 --multi_slice --device cuda > train_resnet_tuned.log 2>&1 &


#Evaluation:
python eval_and_plots.py --multi_slice --device cuda --ckpt results/best_model.pt

#Repo structure:
ele670-alzheimer/
│
├── data/              # MRI data in NIfTI format (.nii.gz)
├── results/           # Metrics, confusion matrices, ROC curves, model checkpoints
├── src/               # Source code (preprocessing, dataset, training, evaluation)
├── main.py            # Training script
├── eval_and_plots.py  # Evaluation and visualization script
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── .gitignore




#GIT: https://github.com/livesorbo/ele670-alzheimer.git 
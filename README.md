# Hippocampus Segmentation

This project implements a 3D U-Net convolutional neural network for automatic segmentation of the hippocampus from MRI scans. The model is trained and tested on medical imaging data stored in NIfTI format.

---

## Project Overview

The hippocampus is a critical brain region involved in memory and cognitive function. Accurate segmentation from MRI scans helps in studying neurodegenerative diseases like Alzheimer's. This project aims to develop a deep learning pipeline for hippocampus segmentation using 3D U-Net.

---

## Features

- 3D U-Net architecture implemented in PyTorch  
- Custom dataset loader for NIfTI medical images  
- Data preprocessing with resizing and normalization  
- Dice loss function for segmentation accuracy  
- Model checkpointing and training visualization  
- Runs on GPU if available  

---

## Prerequisites

- Python 3.7+  
- PyTorch  
- torchvision  
- nibabel  
- scikit-image  
- matplotlib  
- pandas  
- Google Colab (optional)  

You can install the required packages using:

```bash
pip install torch torchvision nibabel scikit-image matplotlib pandas
```


---


## Data Preparation


1. Download the hippocampus dataset (NIfTI files).

2. Organize files in the following structure:

```
hippocampus/
├── nifti/
│   ├── *.nii
│   └── labels/
│       └── *.nii
└── data_splits/
```

3. Run the data splitting script or manually create CSV files listing image and mask paths for training, validation, and testing.

4. Resize images to (128, 128, 128) before feeding into the model.

---



## Usage
Clone the repository

```bash
git clone https://github.com/Jonjakobsen/HippocampusSegmentation.git
cd HippocampusSegmentation
```
Run training

```bash
python train.py
```

---

## Project structure

```
HippocampusSegmentation/
├── data_splits/          # CSV files for train/val/test splits
├── nifti/                # Raw MRI images (.nii files)
│   └── labels/           # Corresponding label masks
├── models/               # Model architecture files (e.g., unet.py)
├── utils/                # Utility functions (data loading, preprocessing)
├── train.py              # Training script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Results

Dice scores:

Average = 0.90
Min = 0.82
Max = 0.92

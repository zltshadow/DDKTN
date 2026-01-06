# Synergizing SAM2 and U-Mamba as Mutual Teachers in a Dual-Branch Dynamic Knowledge Transfer Network for Orbital Segmentation

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-3.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![Mamba](https://img.shields.io/badge/Architecture-Mamba-blueviolet.svg)]()
[![SAM2](https://img.shields.io/badge/Architecture-SAM2-blue.svg)]()

</div>

## 📖 Introduction

This repository contains the official implementation of the paper: **"Synergizing SAM2 and U-Mamba as Mutual Teachers in a Dual-Branch Dynamic Knowledge Transfer Network for Orbital Segmentation"**.

**[Paper Link]** | **[Official GitHub](https://github.com/zltshadow/DDKTN)**

### Abstract
Accurate segmentation of orbital structures in thyroid eye disease (TED) is critical for clinical staging and surgical planning. However, orbital multi-organ segmentation suffers from inherent data scarcity, resulting in substantial domain gaps when directly applying general-purpose segmentation models. [cite_start]These foundation models often lack sufficient exposure to orbital imaging data, producing anatomically implausible yet overconfident predictions [cite: 5-7].

To address these challenges, we propose a **Dual-Branch Dynamic Knowledge Transfer Network (DDKTN)**, which integrates the global semantic generalization capability of **SAM2** with the fine-grained structural modeling strength of **U-Mamba**.

**Key Methodological Innovations:**
1.  Dynamic Feature Interaction (DFI): Enables adaptively controlled inter-branch knowledge transfer, facilitating a progressive transition from SAM2-guided global semantics to U-Mamba-driven structural refinement.
2.  Bidirectional Consensus Confidence (BCC): Identifies consensus and discrepancy regions between the foundation and domain-specific branches, enforcing anatomy-aware consistency via mutual supervision.
3.  Prompt Adaptation Module (PAM): Leverages the anatomical sensitivity of the domain-specific branch to automatically generate structured prompts, thereby injecting reliable anatomical priors into SAM2.

[//]: # (<div align="center">)

[//]: # (  <img src="assets/framework.png" width="800"/>)

[//]: # (  <br>)

[//]: # (  <em>Figure 1: Overview of the proposed DDKTN architecture. The framework synergizes SAM2 and U-Mamba through dynamic feature interaction and mutual supervision.</em>)

[//]: # (</div>)

## 🛠️ Environment Formulation

The code is developed and tested using **Python 3.10+**. Given the dependencies on **Mamba (SSM)** and **SAM 2**, a GPU environment with **CUDA 12.8** support is strictly required.

### 1. Create Virtual Environment

```bash
conda create -n ddktn python=3.13 -y
conda activate ddktn
```

### 2. Install PyTorch (CUDA 12.8)

Install the specific PyTorch version compatible with CUDA 12.8.

```bash
# Install PyTorch 2.7.0 + CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Mamba Dependencies

`mamba-ssm` and `causal-conv1d` require compilation. Ensure `nvcc` (CUDA compiler) is in your path and matches the version used for PyTorch.
```bash
pip install causal-conv1d==1.5.4
pip install mamba-ssm==2.2.6.post3
```

### 4. Install SAM 2

The project relies on the **SAM 2** foundation model. Please install it from source:

```bash
# Clone and install SAM 2
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### 5. Install Core Requirements

Install the remaining dependencies with strict version pinning to ensure reproducibility.

```bash
pip install -r requirements.txt
```

**`requirements.txt` content:**

```text
monai==1.5.1
nnunetv2==2.6.2
transformers==4.57.1
batchgenerators==0.25.1
acvl_utils==0.2.5
dynamic_network_architectures==0.4.2
timm==1.0.21
scikit-image==0.25.2
scikit-learn==1.7.2
simpleitk==2.5.2
tensorboard==2.20.0
torchmetrics==1.8.2
yacs==0.1.8
hydra-core==1.3.2
einops==0.8.1
```

The implementation and model weights will be released after acceptance.

## 📂 Data Preparation

We evaluated our method on two datasets: the private **Orbital Structure Segmentation Dataset (OSSD)** and the public **TOM500** dataset.

> **Privacy Note:** The private **OSSD (CT)** dataset cannot be released due to privacy regulations. However, the code supports the public **TOM500 (MRI)** benchmark or custom datasets following the `nnU-Net` like directory structure.

The dataset should be organized as follows:

```text
DDKTN/
├── data/
│   ├── TOM500_MRI/    # Public Dataset
│   │   ├── imagesTr/      # Training images (.nii.gz)
│   │   ├── labelsTr/      # Training labels (.nii.gz)
│   │   ├── imagesTs/      # Testing images (.nii.gz)
│   │   └── labelsTs/      # Testing labels (.nii.gz)
│   └── OSSD_CT/       # Private Dataset structure
├── checkpoints/
├── logs/
└── ...

```

## 🚀 Usage

### 1. Training

To train the DDKTN model with all components enabled:

```bash
python train.py \
    --root_path ./data/TOM500_MRI \
    --output_dir ./checkpoints/ddktn_mri \
    --model_name DDKTN \
    --batch_size 2 \
    --epochs 100 
```

### 2. Inference

To evaluate the trained model on the test set and calculate metrics:

```bash
python test.py \
    --root_path ./data/TOM500_MRI \
    --checkpoint_path ./checkpoints/ddktn_mri/best_model.pt \
    --save_path ./outputs \
    --save_nii  # Optional: flag to save prediction NIFTI files
```

## 📊 Results

We benchmarked DDKTN against SOTA methods.

### Ablation Study (OSSD Dataset)

The effectiveness of each proposed component is validated as follows:

| Configuration | EB (Dice) | ERM (Dice) | ON (Dice) | **Mean (Dice)** |
| --- | --- | --- | --- | --- |
| Basic | 96.84 | 86.77 | 84.89 | 89.50 |
| Basic + PAM | 98.25 | 87.85 | 86.11 | 90.74 |
| Basic + BCC | 96.94 | 91.08 | 89.52 | 92.52 |
| **DDKTN (Full)** | **98.34** | **91.33** | **91.38** | **93.68** |

## 📜 Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{
  title={Synergizing SAM2 and U-Mamba as Mutual Teachers in a Dual-Branch Dynamic Knowledge Transfer Network for Orbital Segmentation},
  journal={Submitted to IEEE Transactions},
  year={2026}
}

```

## ⚖️ License

This project is released under the [MIT License](https://www.google.com/search?q=LICENSE).

## 🤝 Acknowledgement

We thank the authors of [SAM2](https://github.com/facebookresearch/sam2) and [U-Mamba](https://github.com/bowang-lab/U-Mamba) for making their code publicly available.

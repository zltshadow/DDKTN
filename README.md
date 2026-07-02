# Synergizing SAM2 and U-Mamba as Mutual Teachers in a Dual-Branch Dynamic Knowledge Transfer Network for Orbital Segmentation

## Abstract

Accurate segmentation of orbital structures in thyroid eye disease (TED) is critical for clinical staging and surgical planning. However, orbital multi-organ segmentation remains challenging due to inherent data scarcity, which can lead to substantial domain gaps when directly applying general-purpose foundation segmentation models. These models often lack sufficient exposure to orbital imaging data and may produce anatomically implausible yet overconfident predictions.

To address these challenges, we propose a **Dual-Branch Dynamic Knowledge Transfer Network (DDKTN)**, which integrates the global semantic generalization capability of **SAM2** with the fine-grained structural modeling strength of **U-Mamba**.

### Key Methodological Innovations

1. **Dynamic Feature Interaction (DFI)**Enables adaptively controlled inter-branch knowledge transfer, facilitating a progressive transition from SAM2-guided global semantic stabilization to U-Mamba-driven structural refinement.
2. **Bidirectional Consensus Confidence (BCC)**Identifies consensus and discrepancy regions between the foundation-model branch and the domain-specific branch, enforcing anatomy-aware consistency through mutual supervision.
3. **Prompt Adaptation Module (PAM)**
   Leverages the anatomical sensitivity of the domain-specific branch to automatically generate structured prompts, thereby injecting reliable anatomical priors into SAM2 and reducing the reliance on manual prompts.

## 🛠️ Environment Setup

The code was developed and tested with **Python 3.13+**. Since the framework depends on **Mamba (SSM)** and **SAM2**, a CUDA-enabled GPU environment is required. The recommended setup uses **CUDA 12.8**.

### 1. Create a Virtual Environment

```bash
conda create -n ddktn python=3.13 -y
conda activate ddktn
```

### 2. Install PyTorch with CUDA 12.8

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Mamba Dependencies

`mamba-ssm` and `causal-conv1d` require CUDA compilation. Please make sure that `nvcc` is available in your environment and is compatible with the installed PyTorch CUDA version.

```bash
pip install causal-conv1d==1.5.4
pip install mamba-ssm==2.2.6.post3
```

### 4. Install SAM2

This project relies on the **SAM2** foundation model. Please install SAM2 from the official repository:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### 5. Install Other Requirements

Install the remaining dependencies using:

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
numpy==2.2.6
scipy==1.16.2
Pillow==11.3.0
PyYAML==6.0.3
tqdm==4.67.1
matplotlib==3.10.7
pandas==2.3.3
opencv-python==4.12.0.88
nibabel==5.3.2
SimpleITK==2.5.2
scikit-image==0.25.2
scikit-learn==1.7.2
einops==0.8.1
timm==1.0.21
monai==1.5.1
nnunetv2==2.6.2
batchgenerators==0.25.1
acvl-utils==0.2.5
dynamic-network-architectures==0.4.2
torchmetrics==1.8.2
yacs==0.1.8
hydra-core==1.3.2
transformers==4.57.1
causal-conv1d==1.5.4
mamba-ssm==2.2.6.post3
python-docx==1.2.0
```

## 📂 Data Preparation

In the paper, we evaluate DDKTN on two orbital segmentation datasets: the private **Orbital Structure Segmentation Dataset (OSSD)** and the public **TOM500** dataset.

The recommended dataset structure is:

```text
DDKTN/
├── data/
│   ├── TOM500_MRI/
│   │   ├── imagesTr/      # Training images (.nii.gz)
│   │   ├── labelsTr/      # Training labels (.nii.gz)
│   │   ├── imagesTs/      # Testing images (.nii.gz)
│   │   └── labelsTs/      # Testing labels (.nii.gz)
│   └── OSSD_CT/           # Private dataset structure
├── checkpoints/
├── logs/
└── ...
```

### PASCAL VOC 2012

For the VOC generalization experiment, please first download the standard PASCAL VOC 2012 dataset and organize it as follows:

```text
  DDKTN/
  ├── data/
  │   └── VOCdevkit/
  │       └── VOC2012/
  │           ├── JPEGImages/                 # RGB images (.jpg)
  │           ├── SegmentationClass/          # Semantic labels (.png)
  │           └── ImageSets/
  │               └── Segmentation/
  │                   ├── train.txt
  │                   ├── val.txt
  │                   └── trainval.txt
  ├── checkpoints/
  ├── logs/
  └── ...
```

## 🔄 Loading Model Weights

DDKTN uses two types of checkpoints.

### 1. SAM2 Pretrained Checkpoint

This checkpoint initializes the SAM2 branch:

```bash
--sam2_ckpt checkpoints/sam2.1_hiera_tiny.pt
```

### 2. Trained DDKTN Checkpoint

This checkpoint stores the trained DDKTN model weights and is used for evaluation or inference:

[Google Drive](https://drive.google.com/file/d/1UYFyYLVrX9xEe8AEcPU3aTwONs3If2Y7/view?usp=sharing)

## 🚀 Usage

The current public example provides training and testing scripts on **Pascal VOC** to demonstrate the usability of the pipeline.

### 1. Training

To train the model on Pascal VOC:

```bash
python main.py \
  --data_root ./data/VOCdevkit/VOC2012 \
  --train_split train_aug \
  --batch_size 8 \
  --total_itrs 66100 \
  --val_interval 661 \
  --teacher_weight 0.0 \
  --sam2_repo /path/to/sam2 \
  --sam2_ckpt checkpoints/sam2.1_hiera_tiny.pt \
  --save_dir checkpoints \
  --gpu_id 0
```

### 2. Evaluation

To evaluate a trained model and compute segmentation metrics:

```bash
python main.py \
  --eval_only \
  --ckpt checkpoints/best.pt \
  --data_root ./data/VOCdevkit/VOC2012 \
  --sam2_repo /path/to/sam2 \
  --sam2_ckpt checkpoints/sam2.1_hiera_tiny.pt \
  --save_dir results/ddktn_voc_eval \
  --gpu_id 0
```

### 3. Inference and Visualization

To run inference and save visualization results:

```bash
python test.py \
  --ckpt checkpoints/best.pt \
  --data_root ./data/VOCdevkit/VOC2012 \
  --out_dir results/ddktn_voc513_visual_all \
  --max_samples 0 \
  --gpu_id 0
```

## 📜 Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{zhou2026ddktn,
  title={Synergizing SAM2 and U-Mamba as Mutual Teachers in a Dual-Branch Dynamic Knowledge Transfer Network for Orbital Segmentation},
  author={Zhou, Langtao and Fu, Tianyu and Shi, Jieliang and Shao, Long and Zheng, Te and Ai, Danni and Fan, Jingfan and Xiao, Deqiang and Song, Hong and Wu, Wencan and Yang, Jian},
  journal={Submitted to IEEE Transactions},
  year={2026}
}
```

## ⚖️ License

This project is released under the [MIT License](LICENSE).

## 🤝 Acknowledgement

We sincerely thank the authors of [SAM2](https://github.com/facebookresearch/sam2) and [U-Mamba](https://github.com/bowang-lab/U-Mamba) for making their excellent work publicly available.

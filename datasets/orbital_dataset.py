"""
Medical Image Segmentation Dataset for DDKTN.
Supports standard formats used by Synapse / ACDC / AMOS benchmarks.

Expected directory structure (any of the following):

Format A – pre-sliced .npy files:
    data_root/
        img/
            case0001_slice000.npy
            ...
        label/
            case0001_slice000.npy
            ...

Format B – Synapse-style volumes in a single folder:
    data_root/
        train_vol_h5/          # .npz or .npy, each file = (D, H, W) or (D, H, W, 2)
            case0001.npz       #   keys: 'image', 'label'  (or first/second channel)
            ...
        test_vol_h5/
            case0002.npz
            ...

Format C – nnU-Net style NIfTI:
    data_root/
        imagesTr/
            case_0000.nii.gz
            ...
        labelsTr/
            case_0000.nii.gz
            ...

Reference: https://blog.csdn.net/qq_37541097/article/details/115787033
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from torchvision.transforms import Normalize, Resize, ToTensor
from PIL import Image
import json


class OrbitalDataset(Dataset):
    """Dataset for orbital CT segmentation.

    Expected directory structure:
        data_root/
            imagesTr/
                case_0000.nii.gz    # CT scan
                case_0001.nii.gz
                ...
            labelsTr/
                case_0000.nii.gz    # Segmentation mask
                case_0001.nii.gz
                ...

    Or for 2D slice-based loading:
        data_root/
            images/
                case_0000_slice000.npy
                ...
            labels/
                case_0000_slice000.npy
                ...

    Args:
        data_root: Root directory of the dataset
        split: 'train', 'val', or 'test'
        mode: '3d' for volume-based or '2d' for slice-based loading
        target_size: Target spatial size for resizing (H, W)
        num_classes: Number of segmentation classes
        transform: Optional additional transforms
        sam2_transform: Transform for SAM2 branch (resize to 1024 + ImageNet normalize)
        slice_axis: Axis along which to extract 2D slices (for mode='3d')
        augment: Whether to apply data augmentation
    """
    def __init__(self, data_root, split='train', mode='2d',
                 target_size=(256, 256), num_classes=4,
                 transform=None, sam2_transform=None,
                 slice_axis=2, augment=False,
                 fold=None, num_folds=None,
                 val_ratio=0.2,
                 window_center=None, window_width=None):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.target_size = target_size
        self.num_classes = num_classes
        self.transform = transform
        self.sam2_transform = sam2_transform
        self.slice_axis = slice_axis
        self.augment = augment and (split == 'train')

        # SAM2 preprocessing
        self.sam2_mean = [0.485, 0.456, 0.406]
        self.sam2_std = [0.229, 0.224, 0.225]
        self.sam2_size = 1024

        # CT windowing (optional)
        self.window_center = window_center
        self.window_width = window_width
        self.val_ratio = val_ratio

        # Load file list
        self.samples = self._load_file_list(fold, num_folds)

    # ── file discovery ──────────────────────────────────────

    def _load_file_list(self, fold, num_folds):
        """Load and split data files.  Supports three directory layouts."""
        samples = []

        # ── Format A: pre-sliced .npy / .npz (img/ + label/) ──
        img_dir_a = os.path.join(self.data_root, 'img')
        lbl_dir_a = os.path.join(self.data_root, 'label')
        if os.path.isdir(img_dir_a) and os.path.isdir(lbl_dir_a):
            for ext in ('*.npy', '*.npz'):
                for img_path in sorted(glob.glob(os.path.join(img_dir_a, ext))):
                    fname = os.path.basename(img_path)
                    lbl_path = os.path.join(lbl_dir_a, fname)
                    if os.path.exists(lbl_path):
                        samples.append({
                            'image': img_path, 'label': lbl_path,
                            'case_id': os.path.splitext(fname)[0],
                        })
            return self._cv_split(samples, fold, num_folds)

        # ── Format B: Synapse-style volumes (train_vol_h5/ + test_vol_h5/) ──
        for sub in ('train_vol_h5', 'test_vol_h5', 'train_npz', 'test_npz'):
            vol_dir = os.path.join(self.data_root, sub)
            if not os.path.isdir(vol_dir):
                continue
            for ext in ('*.npz', '*.npy', '*.h5', '*.hdf5'):
                for vol_path in sorted(glob.glob(os.path.join(vol_dir, ext))):
                    case_id = os.path.splitext(os.path.basename(vol_path))[0]
                    samples.append({
                        'image': vol_path, 'label': vol_path,
                        'case_id': case_id,
                    })

        if samples:
            return self._cv_split(samples, fold, num_folds)

        # ── Format C: nnU-Net style NIfTI (imagesTr/ + labelsTr/) ──
        for img_sub, lbl_sub in [('imagesTr', 'labelsTr'), ('images', 'labels')]:
            img_dir = os.path.join(self.data_root, img_sub)
            lbl_dir = os.path.join(self.data_root, lbl_sub)
            if os.path.isdir(img_dir):
                for img_path in sorted(glob.glob(os.path.join(img_dir, '*.nii.gz'))):
                    case_id = os.path.basename(img_path).replace('.nii.gz', '')
                    lbl_path = os.path.join(lbl_dir, case_id + '.nii.gz')
                    if os.path.exists(lbl_path):
                        samples.append({
                            'image': img_path, 'label': lbl_path,
                            'case_id': case_id,
                        })
                break

        return self._cv_split(samples, fold, num_folds)

    def _cv_split(self, samples, fold, num_folds):
        """Split data into train/val/test.

        If fold and num_folds are provided (num_folds > 1), performs cross-validation.
        Otherwise, performs a simple train/val split based on val_ratio.
        """
        if not samples:
            return samples

        # Simple train/val split (no cross-validation)
        if fold is None or num_folds is None or num_folds <= 1:
            n = len(samples)
            val_size = max(int(n * self.val_ratio), 1)
            if self.split == 'train':
                return samples[val_size:]
            elif self.split == 'val':
                return samples[:val_size]
            return samples  # 'test' uses all

        # Cross-validation split
        n = len(samples)
        fold_size = max(n // num_folds, 1)
        val_start = fold * fold_size
        val_end = min(val_start + fold_size, n) if fold < num_folds - 1 else n

        if self.split == 'train':
            return samples[:val_start] + samples[val_end:]
        elif self.split == 'val':
            return samples[val_start:val_end]
        return samples   # 'test' uses all

    def __len__(self):
        return len(self.samples)

    def _load_nifti(self, path):
        """Load a NIfTI file and return as numpy array."""
        nii = nib.load(path)
        data = nii.get_fdata().astype(np.float32)
        return data

    def _apply_windowing(self, img):
        """Apply CT windowing if window_center/width are set."""
        if self.window_center is None or self.window_width is None:
            return img
        lower = self.window_center - self.window_width / 2
        upper = self.window_center + self.window_width / 2
        img = np.clip(img, lower, upper)
        return img

    def _load_sample_2d(self, idx):
        """Load a single 2D sample (supports .npy, .npz, and .nii.gz)."""
        sample = self.samples[idx]
        img_path = sample['image']
        lbl_path = sample['label']
        ext = os.path.splitext(img_path)[1]

        # ── Load image ──
        if ext == '.npz':
            data = np.load(img_path)
            if 'image' in data:
                img = data['image'].astype(np.float32)
                lbl = data['label'].astype(np.int64)
            else:
                # Assume first array is image, second is label
                keys = list(data.keys())
                img = data[keys[0]].astype(np.float32)
                lbl = data[keys[1]].astype(np.int64) if len(keys) > 1 else np.zeros_like(img, dtype=np.int64)
        elif ext in ('.nii', '.gz'):
            img = self._load_nifti(img_path)
            lbl = self._load_nifti(lbl_path).astype(np.int64)
            # Extract a 2D slice from 3D volume
            num_slices = img.shape[self.slice_axis]
            slice_idx = np.random.randint(0, num_slices) if self.augment else num_slices // 2
            img = np.take(img, slice_idx, axis=self.slice_axis)
            lbl = np.take(lbl, slice_idx, axis=self.slice_axis)
        else:
            img = np.load(img_path).astype(np.float32)
            lbl = np.load(lbl_path).astype(np.int64)

        # Ensure 2D
        if img.ndim == 3:
            img = img[..., 0]
        if lbl.ndim == 3:
            lbl = lbl[..., 0]

        return img, lbl, sample['case_id']

    def _load_sample_3d(self, idx):
        """Load a 3D volume and extract a random slice."""
        return self._load_sample_2d(idx)  # unified in _load_sample_2d

    def _normalize_image(self, img):
        """Normalize image to [0, 1] range (with optional CT windowing)."""
        img = self._apply_windowing(img)
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        return img

    def _prepare_umamba_input(self, img):
        """Prepare input for U-Mamba branch: resize to target_size.

        Args:
            img: (H, W) numpy array, normalized to [0, 1]
        Returns:
            tensor: (1, H_target, W_target)
        """
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img_pil)).float().unsqueeze(0) / 255.0
        return tensor

    def _prepare_sam2_input(self, img):
        """Prepare input for SAM2 branch: resize to 1024 + ImageNet normalize.

        Args:
            img: (H, W) numpy array, normalized to [0, 1]
        Returns:
            tensor: (3, 1024, 1024) normalized for ImageNet
        """
        # Convert to 3-channel
        img_3ch = np.stack([img, img, img], axis=-1)
        img_pil = Image.fromarray((img_3ch * 255).astype(np.uint8))
        img_pil = img_pil.resize((self.sam2_size, self.sam2_size), Image.BILINEAR)

        tensor = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1) / 255.0
        normalize = Normalize(self.sam2_mean, self.sam2_std)
        tensor = normalize(tensor)
        return tensor

    def _augment(self, img, lbl):
        """Apply simple data augmentation."""
        if np.random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            lbl = np.flip(lbl, axis=1).copy()
        if np.random.random() > 0.5:
            img = np.flip(img, axis=0).copy()
            lbl = np.flip(lbl, axis=0).copy()
        # Random brightness/contrast
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)
        return img, lbl

    def __getitem__(self, idx):
        # Load data (unified: handles .npy, .npz, .nii.gz)
        img, lbl, case_id = self._load_sample_2d(idx)

        # Normalize (with optional CT windowing)
        img = self._normalize_image(img)

        # Augmentation
        if self.augment:
            img, lbl = self._augment(img, lbl)

        # Resize labels to target_size
        lbl_pil = Image.fromarray(lbl.astype(np.uint8))
        lbl_resized = np.array(lbl_pil.resize(
            (self.target_size[1], self.target_size[0]), Image.NEAREST
        )).astype(np.int64)

        # Prepare branch inputs
        img_um = self._prepare_umamba_input(img)     # (1, H, W)
        img_sam2 = self._prepare_sam2_input(img)      # (3, 1024, 1024)
        lbl_tensor = torch.from_numpy(lbl_resized)    # (H, W)

        sample = {
            'image_um': img_um,
            'image_sam2': img_sam2,
            'label': lbl_tensor,
            'case_id': case_id,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def create_dataloader(data_root, batch_size=4, split='train', mode='2d',
                      target_size=(256, 256), num_classes=4,
                      num_workers=4, fold=None, num_folds=None,
                      val_ratio=0.2, augment=True,
                      window_center=None, window_width=None):
    """Create a DataLoader for medical image segmentation.

    Args:
        data_root: Root directory of the dataset
        batch_size: Batch size
        split: 'train', 'val', or 'test'
        mode: '2d' or '3d' (kept for backward compatibility; always loads 2D)
        target_size: Target spatial size
        num_classes: Number of classes
        num_workers: Number of data loading workers
        fold: Cross-validation fold (None for simple split)
        num_folds: Total number of folds (None for simple split)
        val_ratio: Validation ratio for simple split (default: 0.2)
        augment: Whether to apply augmentation (train only)
        window_center: CT window center (optional)
        window_width: CT window width (optional)
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader

    dataset = OrbitalDataset(
        data_root=data_root,
        split=split,
        mode=mode,
        target_size=target_size,
        num_classes=num_classes,
        fold=fold,
        num_folds=num_folds,
        val_ratio=val_ratio,
        augment=augment,
        window_center=window_center,
        window_width=window_width,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )

    return loader

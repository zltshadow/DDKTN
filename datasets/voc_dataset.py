"""
PASCAL VOC 2012 Dataset for DDKTN.

Supports two loading modes:
  1. Pre-converted .npy (from scripts/prepare_voc2012.py):
        data_root/
            img/         *.npy  (H, W, 3) uint8
            label/       *.npy  (H, W)    int64  (255=void)
            train.txt / val.txt / trainval.txt

  2. Raw VOC2012 directory:
        data_root/
            JPEGImages/           *.jpg
            SegmentationClass/    *.png
            ImageSets/Segmentation/
                train.txt / val.txt

21 classes: 0=background, 1-20=objects.  Pixel 255 = void/ignore.

Reference: https://blog.csdn.net/qq_37541097/article/details/115787033
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from PIL import Image


# VOC2012 class definitions
VOC_CLASSES = [
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
VOC_NUM_CLASSES = 21
VOC_IGNORE_INDEX = 255

# VOC color palette for saving predictions (21 classes + void)
VOC_PALETTE = [
    0, 0, 0,        # 0 background
    128, 0, 0,      # 1 aeroplane
    0, 128, 0,      # 2 bicycle
    128, 128, 0,    # 3 bird
    0, 0, 128,      # 4 boat
    128, 0, 128,    # 5 bottle
    0, 128, 128,    # 6 bus
    128, 128, 128,  # 7 car
    64, 0, 0,       # 8 cat
    192, 0, 0,      # 9 chair
    64, 128, 0,     # 10 cow
    192, 128, 0,    # 11 diningtable
    64, 0, 128,     # 12 dog
    192, 0, 128,    # 13 horse
    64, 128, 128,   # 14 motorbike
    192, 128, 128,  # 15 person
    0, 64, 0,       # 16 pottedplant
    128, 64, 0,     # 17 sheep
    0, 192, 0,      # 18 sofa
    128, 192, 0,    # 19 train
    0, 64, 128,     # 20 tvmonitor
    224, 224, 192,  # 21 boundary/void
]


class VOC2012Dataset(Dataset):
    """PASCAL VOC 2012 semantic segmentation dataset.

    Args:
        data_root: Path to pre-converted data (img/ + label/) or raw VOC2012
        split: 'train', 'train_aug', 'val', 'trainval', or 'test'
        target_size: (H, W) for U-Mamba branch input
        num_classes: Number of segmentation classes (default 21)
        augment: Apply random flip, scale, color jitter (train only)
    """
    def __init__(self, data_root, split='train',
                 target_size=(256, 256), num_classes=VOC_NUM_CLASSES,
                 augment=False):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.target_size = target_size
        self.num_classes = num_classes
        self.augment = augment and (split in ('train', 'trainval', 'train_aug'))

        # SAM2 preprocessing
        self.sam2_mean = [0.485, 0.456, 0.406]
        self.sam2_std  = [0.229, 0.224, 0.225]
        self.sam2_size = 1024

        self.samples = self._load_file_list()

    # ── file discovery ──────────────────────────────────────

    def _load_file_list(self):
        """Auto-detect pre-converted .npy or raw VOC2012 layout."""
        samples = []

        # ── Pre-converted .npy ──
        img_dir = os.path.join(self.data_root, 'img')
        lbl_dir = os.path.join(self.data_root, 'label')
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            # Read split list if available
            split_txt = os.path.join(self.data_root, self.split + '.txt')
            if os.path.exists(split_txt):
                with open(split_txt) as f:
                    ids = [l.strip() for l in f if l.strip()]
                for sid in ids:
                    # Try multiple extensions: .npy, .jpg, .jpeg, .png
                    ip = lp = None
                    for ext in ('.npy', '.jpg', '.jpeg', '.png'):
                        p = os.path.join(img_dir, sid + ext)
                        if os.path.exists(p):
                            ip = p
                            break
                    for ext in ('.npy', '.png', '.jpg'):
                        p = os.path.join(lbl_dir, sid + ext)
                        if os.path.exists(p):
                            lp = p
                            break
                    if ip and lp:
                        samples.append({'image': ip, 'label': lp, 'case_id': sid})
            elif self.split in ('all', 'test'):
                # Explicit all/test mode may use every image when no split file exists.
                import glob
                for ip in sorted(glob.glob(os.path.join(img_dir, '*.npy'))
                                 + glob.glob(os.path.join(img_dir, '*.jpg'))
                                 + glob.glob(os.path.join(img_dir, '*.jpeg'))
                                 + glob.glob(os.path.join(img_dir, '*.png'))):
                    fname = os.path.basename(ip)
                    stem = os.path.splitext(fname)[0]
                    lp = None
                    for ext in ('.npy', '.png', '.jpg'):
                        p = os.path.join(lbl_dir, stem + ext)
                        if os.path.exists(p):
                            lp = p
                            break
                    if lp:
                        samples.append({'image': ip, 'label': lp, 'case_id': stem})
            if samples:
                return samples

        # ── Raw VOC2012 ──
        jpeg_dir = os.path.join(self.data_root, 'JPEGImages')
        cls_dir  = os.path.join(self.data_root, 'SegmentationClass')
        aug_cls_dir = os.path.join(self.data_root, 'SegmentationClassAug')
        mask_dir = aug_cls_dir if self.split == 'train_aug' and os.path.isdir(aug_cls_dir) else cls_dir
        split_txt = os.path.join(self.data_root, 'ImageSets', 'Segmentation',
                                 self.split + '.txt')
        if os.path.isdir(jpeg_dir) and os.path.exists(split_txt):
            with open(split_txt) as f:
                ids = [l.strip() for l in f if l.strip()]
            for sid in ids:
                ip = os.path.join(jpeg_dir, sid + '.jpg')
                lp = os.path.join(mask_dir, sid + '.png')
                if not os.path.exists(lp) and mask_dir != cls_dir:
                    lp = os.path.join(cls_dir, sid + '.png')
                if os.path.exists(ip) and os.path.exists(lp):
                    samples.append({'image': ip, 'label': lp, 'case_id': sid})

        return samples

    def __len__(self):
        return len(self.samples)

    # ── loading ─────────────────────────────────────────────

    def _load_sample(self, idx):
        """Load image as (H, W, 3) float32 [0,1] and label as (H, W) int64."""
        sample = self.samples[idx]
        img_path, lbl_path = sample['image'], sample['label']

        ext = os.path.splitext(img_path)[1].lower()

        if ext == '.npy':
            img = np.load(img_path).astype(np.float32)
            lbl = np.load(lbl_path).astype(np.int64)
            # If label has extra dims, squeeze
            if lbl.ndim == 3:
                lbl = lbl[..., 0]
        elif ext in ('.jpg', '.jpeg', '.png'):
            img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)
            lbl = np.array(Image.open(lbl_path)).astype(np.int64)
        else:
            raise ValueError(f'Unsupported image format: {ext}')

        # Normalize image to [0, 1]
        if img.max() > 1.0:
            img /= 255.0

        # Ensure 3-channel
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        return img, lbl, sample['case_id']

    # ── augmentation ────────────────────────────────────────

    def _augment(self, img, lbl):
        """Random horizontal flip + scale jitter + color jitter."""
        H, W = img.shape[:2]

        # Random horizontal flip
        if np.random.random() > 0.5:
            img = img[:, ::-1].copy()
            lbl = lbl[:, ::-1].copy()

        # Random scale jitter (0.5 – 2.0), then crop to target_size
        scale = np.random.uniform(0.5, 2.0)
        new_h, new_w = int(H * scale), int(W * scale)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        lbl_pil = Image.fromarray(lbl.astype(np.uint8))
        img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
        lbl_pil = lbl_pil.resize((new_w, new_h), Image.NEAREST)
        img = np.array(img_pil).astype(np.float32) / 255.0
        lbl = np.array(lbl_pil).astype(np.int64)

        # Pad with ignore labels if scaled image is smaller than crop, then crop.
        th, tw = self.target_size
        pad_h = max(th - new_h, 0)
        pad_w = max(tw - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            top = np.random.randint(0, pad_h + 1) if pad_h > 0 else 0
            left = np.random.randint(0, pad_w + 1) if pad_w > 0 else 0
            bottom = pad_h - top
            right = pad_w - left
            img = np.pad(
                img, ((top, bottom), (left, right), (0, 0)),
                mode='constant', constant_values=0,
            )
            lbl = np.pad(
                lbl, ((top, bottom), (left, right)),
                mode='constant', constant_values=VOC_IGNORE_INDEX,
            )
            new_h, new_w = img.shape[:2]

        y = np.random.randint(0, new_h - th + 1)
        x = np.random.randint(0, new_w - tw + 1)
        for _ in range(10):
            cand_y = np.random.randint(0, new_h - th + 1)
            cand_x = np.random.randint(0, new_w - tw + 1)
            cand_lbl = lbl[cand_y:cand_y+th, cand_x:cand_x+tw]
            valid = cand_lbl != VOC_IGNORE_INDEX
            if valid.any():
                labels, counts = np.unique(cand_lbl[valid], return_counts=True)
                if len(labels) > 1 and counts.max() / counts.sum() < 0.75:
                    y, x = cand_y, cand_x
                    break
        img = img[y:y+th, x:x+tw]
        lbl = lbl[y:y+th, x:x+tw]

        # Color jitter (brightness / contrast)
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.75, 1.25)
            img = np.clip(img * factor, 0, 1)

        return img, lbl

    # ── input preparation ───────────────────────────────────

    def _prepare_umamba_input(self, img):
        """(H, W, 3) float [0,1] → (3, H_t, W_t) tensor.

        VOC images are RGB.  We resize directly and keep 3 channels
        so the U-Mamba encoder should be built with input_channels=3
        when training on VOC.  Apply ImageNet normalisation (same as
        DeepLabV3Plus and SAM2 branch) so the model receives standard
        input statistics.
        """
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize(
            (self.target_size[1], self.target_size[0]), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1) / 255.0
        tensor = Normalize(self.sam2_mean, self.sam2_std)(tensor)
        return tensor   # (3, H, W)

    def _prepare_sam2_input(self, img):
        """(H, W, 3) float [0,1] → (3, 1024, 1024) ImageNet-normalised tensor."""
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((self.sam2_size, self.sam2_size), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1) / 255.0
        tensor = Normalize(self.sam2_mean, self.sam2_std)(tensor)
        return tensor

    # ── main entry ──────────────────────────────────────────

    def __getitem__(self, idx):
        img, lbl, case_id = self._load_sample(idx)   # img: (H,W,3) [0,1], lbl: (H,W)

        if self.augment:
            img, lbl = self._augment(img, lbl)

        # Resize label to target_size
        lbl_pil = Image.fromarray(lbl.astype(np.uint8))
        lbl_resized = np.array(
            lbl_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        ).astype(np.int64)

        img_um   = self._prepare_umamba_input(img)    # (3, H, W)
        img_sam2 = self._prepare_sam2_input(img)       # (3, 1024, 1024)

        # Convert VOC void (255) → -1 so CrossEntropyLoss ignores it
        lbl_tensor = torch.from_numpy(lbl_resized.astype(np.int64))
        lbl_tensor[lbl_tensor == 255] = -1

        return {
            'image_um':   img_um,
            'image_sam2': img_sam2,
            'label':      lbl_tensor,
            'case_id':    case_id,
        }


def create_voc_dataloader(data_root, batch_size=4, split='train',
                          target_size=(256, 256), num_classes=VOC_NUM_CLASSES,
                          num_workers=4, augment=True):
    """Create a DataLoader for VOC2012.

    Args:
        data_root: Path to pre-converted or raw VOC2012
        batch_size: Batch size
        split: 'train', 'val', 'trainval'
        target_size: (H, W) for U-Mamba input
        num_classes: 21 for VOC2012
        num_workers: Dataloader workers
        augment: Augment on train splits
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader

    dataset = VOC2012Dataset(
        data_root=data_root,
        split=split,
        target_size=target_size,
        num_classes=num_classes,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split in ('train', 'trainval', 'train_aug')),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split in ('train', 'trainval', 'train_aug')),
    )

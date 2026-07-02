"""
Prepare PASCAL VOC 2012 dataset for DDKTN training.

Converts JPEG + palette PNG into per-sample .npy files.
VOC2012 has 21 classes (0=background, 1-20=object categories), pixel 255 = void/ignore.

Usage:
    python scripts/prepare_voc2012.py \
        --voc_root /path/to/VOCdevkit/VOC2012 \
        --output_dir ./data/voc2012 \
        --use_augmented               # optionally merge SBD augmentation

Input layout (standard VOC2012):
    VOC2012/
        JPEGImages/           *.jpg
        SegmentationClass/    *.png   (palette mode, 0-20, 255=void)
        ImageSets/Segmentation/
            train.txt / val.txt / trainval.txt

Output layout:
    output_dir/
        img/
            2007_000027.npy       # (H, W, 3) uint8  RGB
            ...
        label/
            2007_000027.npy       # (H, W)    int64   class indices (255 preserved)
            ...
        train.txt / val.txt / trainval.txt   # split lists

Reference: https://blog.csdn.net/qq_37541097/article/details/115787033
"""
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


# ── VOC2012 class names (21 including background) ──────────
VOC_CLASSES = [
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
NUM_CLASSES = 21


def read_list(txt_path):
    """Read a split file and return list of image IDs."""
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def convert_sample(img_id, voc_root, out_img_dir, out_lbl_dir, fmt='jpg'):
    """Convert one VOC sample.

    fmt='npy'  → save as .npy (fast loading, not human-readable)
    fmt='jpg'  → copy original .jpg + .png (easy to preview)
    """
    img_path = os.path.join(voc_root, 'JPEGImages', img_id + '.jpg')
    lbl_path = os.path.join(voc_root, 'SegmentationClass', img_id + '.png')

    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        return False

    if fmt == 'npy':
        img = np.array(Image.open(img_path).convert('RGB'))
        lbl = np.array(Image.open(lbl_path))
        np.save(os.path.join(out_img_dir, img_id + '.npy'), img)
        np.save(os.path.join(out_lbl_dir, img_id + '.npy'), lbl.astype(np.int64))
    else:
        import shutil
        # Copy jpg as-is
        shutil.copy2(img_path, os.path.join(out_img_dir, img_id + '.jpg'))
        # Copy label png as-is (palette mode preserved)
        shutil.copy2(lbl_path, os.path.join(out_lbl_dir, img_id + '.png'))
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare VOC2012 for DDKTN')
    parser.add_argument('--voc_root', type=str, required=True,
                        help='Path to VOC2012 directory (contains JPEGImages/ etc.)')
    parser.add_argument('--output_dir', type=str, default='./data/voc2012',
                        help='Output directory')
    parser.add_argument('--use_augmented', action='store_true',
                        help='Merge SBD augmented data (requires SegmentationClassAug/)')
    parser.add_argument('--format', type=str, default='jpg',
                        choices=['npy', 'jpg'],
                        help='Output format: jpg (copy originals, easy to browse) or npy')
    args = parser.parse_args()

    voc_root = args.voc_root
    out_img_dir = os.path.join(args.output_dir, 'img')
    out_lbl_dir = os.path.join(args.output_dir, 'label')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Read split lists
    split_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')
    splits = {}
    for split_name in ('train', 'val', 'trainval'):
        txt = os.path.join(split_dir, split_name + '.txt')
        if os.path.exists(txt):
            splits[split_name] = read_list(txt)

    # If using augmented data, override train with trainaug
    if args.use_augmented:
        aug_lbl_dir = os.path.join(voc_root, 'SegmentationClassAug')
        if os.path.isdir(aug_lbl_dir):
            print(f'Using augmented labels from {aug_lbl_dir}')
            # SBD augmentation provides additional training images
            aug_ids = [f.replace('.png', '') for f in sorted(os.listdir(aug_lbl_dir))
                       if f.endswith('.png')]
            if 'train' in splits:
                splits['train'] = list(set(splits['train'] + aug_ids))
                splits['train'].sort()
            # Also update SegmentationClass symlink for augmented labels
            orig_lbl_dir = os.path.join(voc_root, 'SegmentationClass')
            for fn in os.listdir(aug_lbl_dir):
                src = os.path.join(aug_lbl_dir, fn)
                dst = os.path.join(orig_lbl_dir, fn)
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)
        else:
            print(f'WARNING: {aug_lbl_dir} not found, skipping augmentation')

    # Collect all unique IDs
    all_ids = sorted(set(sum(splits.values(), [])))
    print(f'Total unique samples: {len(all_ids)}')
    for name, ids in splits.items():
        print(f'  {name}: {len(ids)}')

    # Convert
    success = 0
    for img_id in tqdm(all_ids, desc='Converting'):
        if convert_sample(img_id, voc_root, out_img_dir, out_lbl_dir, fmt=args.format):
            success += 1

    # Save split lists
    for name, ids in splits.items():
        with open(os.path.join(args.output_dir, name + '.txt'), 'w') as f:
            f.write('\n'.join(ids) + '\n')

    print(f'Done! {success}/{len(all_ids)} samples saved to {args.output_dir}/')
    print(f'  Format: {args.format}')
    print(f'  img/   -> {len(os.listdir(out_img_dir))} files')
    print(f'  label/ -> {len(os.listdir(out_lbl_dir))} files')
    print(f'  Classes: {NUM_CLASSES} (0=background, 1-20=objects, 255=void)')


if __name__ == '__main__':
    main()

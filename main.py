"""
VOC2012 training entrypoint for DDKTN with mIoU logging.

This keeps VOC semantic-segmentation metrics while moving the optimizer
and regularization closer to the DDKTN manuscript: SAM-side modules use
AdamW + cosine decay, U-Mamba/PAM modules use SGD + Nesterov + PolyLR,
and BCC/KD can be enabled for mutual-teacher training on VOC.
"""

import argparse
import csv
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from PIL import Image
from torchvision.transforms import Normalize

from datasets.voc_dataset import VOC2012Dataset, create_voc_dataloader
from models.DDKTN import DDKTN
from models.BCC import BCCModule
from train_umamba_deeplab import PolyLR, SegmentationLoss, StreamSegMetrics


class VOC513CenterCropValDataset(VOC2012Dataset):
    """VOC validation transform matching DeepLab README: Resize(513), CenterCrop(513)."""
    def __getitem__(self, idx):
        img, lbl, case_id = self._load_sample(idx)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        lbl_pil = Image.fromarray(lbl.astype(np.uint8))
        width, height = img_pil.size
        if width < height:
            resized_width = self.target_size[1]
            resized_height = int(self.target_size[1] * height / width)
        else:
            resized_height = self.target_size[0]
            resized_width = int(self.target_size[0] * width / height)
        img_pil = img_pil.resize((resized_width, resized_height), Image.BILINEAR)
        lbl_pil = lbl_pil.resize((resized_width, resized_height), Image.NEAREST)
        left = int(round((resized_width - self.target_size[1]) / 2.0))
        top = int(round((resized_height - self.target_size[0]) / 2.0))
        right = left + self.target_size[1]
        bottom = top + self.target_size[0]
        img = np.array(img_pil.crop((left, top, right, bottom))).astype(np.float32) / 255.0
        lbl = np.array(lbl_pil.crop((left, top, right, bottom))).astype(np.int64)

        img_um = self._prepare_umamba_input(img)
        img_sam2 = self._prepare_sam2_input(img)
        lbl_tensor = torch.from_numpy(lbl)
        lbl_tensor[lbl_tensor == 255] = -1
        return {
            "image_um": img_um,
            "image_sam2": img_sam2,
            "label": lbl_tensor,
            "case_id": case_id,
        }


def create_official_voc513_val_dataloader(data_root, batch_size, target_size, num_classes, num_workers):
    dataset = VOC513CenterCropValDataset(
        data_root=data_root, split="val", target_size=target_size,
        num_classes=num_classes, augment=False,
    )
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )


def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("train_ddktn_voc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(os.path.join(save_dir, "train.log"), mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    loss_csv = os.path.join(save_dir, "loss.csv")
    if not os.path.exists(loss_csv):
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "epoch", "loss", "main_loss", "lr_sam", "lr_mamba"])

    val_csv = os.path.join(save_dir, "val.csv")
    if not os.path.exists(val_csv):
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "epoch", "mean_iou", "overall_acc", "mean_acc"])
    return logger


def build_sam2(sam2_repo, sam2_config, sam2_ckpt, device):
    if sam2_repo not in sys.path:
        sys.path.insert(0, sam2_repo)
    from sam2.build_sam import build_sam2

    return build_sam2(
        config_file=sam2_config,
        ckpt_path=sam2_ckpt if os.path.exists(sam2_ckpt) else None,
        device=str(device),
        mode="eval",
    )


def build_teacher(opts, device, logger):
    needs_teacher = (opts.teacher_weight > 0 or
                     getattr(opts, "eval_teacher_fusion_weight", 0.0) > 0 or
                     bool(getattr(opts, "eval_teacher_fusion_sweep", None)))
    if not opts.teacher_ckpt or not needs_teacher:
        return None
    if not os.path.isfile(opts.teacher_ckpt):
        logger.warning("Teacher checkpoint not found: %s", opts.teacher_ckpt)
        return None

    deeplab_repo = opts.deeplab_repo
    if deeplab_repo not in sys.path:
        sys.path.insert(0, deeplab_repo)
    import network

    model_fn = network.modeling.__dict__[opts.teacher_model]
    try:
        teacher = model_fn(
            num_classes=opts.num_classes,
            output_stride=opts.teacher_output_stride,
            pretrained_backbone=False,
        )
    except TypeError:
        teacher = model_fn(
            num_classes=opts.num_classes,
            output_stride=opts.teacher_output_stride,
        )

    ckpt = torch.load(opts.teacher_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    teacher.to(device).eval()
    for p_ in teacher.parameters():
        p_.requires_grad = False
    logger.info(
        "Loaded teacher: %s from %s (missing=%d, unexpected=%d)",
        opts.teacher_model, opts.teacher_ckpt, len(missing), len(unexpected),
    )
    return teacher


def distillation_loss(student_logits, teacher_logits, labels, temperature):
    if teacher_logits.shape[2:] != student_logits.shape[2:]:
        teacher_logits = F.interpolate(
            teacher_logits, size=student_logits.shape[2:],
            mode="bilinear", align_corners=False,
        )
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    per_pixel = F.kl_div(log_p, q, reduction="none").sum(dim=1)
    valid = labels >= 0
    if valid.any():
        per_pixel = per_pixel[valid]
    return per_pixel.mean() * (temperature ** 2)


def build_model(opts, device):
    sam2_model = build_sam2(opts.sam2_repo, opts.sam2_config, opts.sam2_ckpt, device)
    model = DDKTN(
        num_classes=opts.num_classes,
        input_channels_um=3,
        input_size=(opts.crop_size, opts.crop_size),
        sam2_model=sam2_model,
        um_features_per_stage=tuple(opts.features),
        um_strides=tuple((s, s) for s in opts.strides),
        um_n_blocks=tuple(opts.n_blocks),
        sam2_backbone_channels=tuple([256] * opts.dfi_stages),
        dfi_align_channels=opts.dfi_align_channels,
        dfi_num_heads=opts.dfi_num_heads,
        pam_tau=opts.pam_tau,
        pam_bilateral=False,
        freeze_sam2_backbone=opts.freeze_sam2_all,
    )
    if opts.sam2_train_last_blocks > 0:
        for p in model.sam2.parameters():
            p.requires_grad = False
        blocks = model.sam2.image_encoder.trunk.blocks
        start = max(len(blocks) - opts.sam2_train_last_blocks, 0)
        prefixes = tuple(f"trunk.blocks.{idx}." for idx in range(start, len(blocks)))
        for name, p in model.sam2.image_encoder.named_parameters():
            if name.startswith(prefixes) or name.startswith("neck."):
                p.requires_grad = True
        # DDKTN uses PAM prompts in the SAM2 prompt encoder and consumes the
        # mask decoder output in its semantic SAM branch, so both must adapt.
        for module in (model.sam2.prompt_encoder, model.sam2.mask_decoder):
            for p in module.parameters():
                p.requires_grad = True
    elif opts.freeze_sam2_all:
        for p in model.sam2.parameters():
            p.requires_grad = False
    return model.to(device)


def forward_model(model, images_um, images_sam2, cur_itrs, total_itrs):
    outputs = model(images_um, images_sam2, t=cur_itrs, T_max=total_itrs)
    logits_fused, logits_sam, logits_um, prob_um, bcc_info = outputs
    target_size = images_um.shape[-2:]
    if logits_fused.shape[2:] != target_size:
        logits_fused = F.interpolate(
            logits_fused, size=target_size, mode="bilinear", align_corners=False
        )
    if logits_sam.shape[2:] != target_size:
        logits_sam = F.interpolate(
            logits_sam, size=target_size, mode="bilinear", align_corners=False
        )
    if logits_um.shape[2:] != target_size:
        logits_um = F.interpolate(
            logits_um, size=target_size, mode="bilinear", align_corners=False
        )
    return logits_fused, logits_sam, logits_um, prob_um, bcc_info


@torch.no_grad()
def validate(model, loader, device, metrics, total_itrs, max_eval=0,
             teacher=None, teacher_fusion_weight=0.0):
    model.eval()
    if teacher is not None:
        teacher.eval()
    metrics.reset()
    for idx, batch in enumerate(loader):
        if max_eval and idx >= max_eval:
            break
        images_um = batch["image_um"].to(device, dtype=torch.float32)
        images_sam2 = batch["image_sam2"].to(device, dtype=torch.float32)
        labels = batch["label"].to(device, dtype=torch.long)
        logits, _, _, _, _ = forward_model(model, images_um, images_sam2, total_itrs, total_itrs)
        if teacher is not None and teacher_fusion_weight > 0:
            teacher_logits = teacher(images_um)
            if teacher_logits.shape[2:] != logits.shape[2:]:
                teacher_logits = F.interpolate(
                    teacher_logits, size=logits.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            logits = (1.0 - teacher_fusion_weight) * logits + teacher_fusion_weight * teacher_logits
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        targets = np.where(targets < 0, 255, targets)
        metrics.update(targets, preds)
    return metrics.get_results()


@torch.no_grad()
def validate_fusion_sweep(model, loader, device, total_itrs, fusion_weights, num_classes,
                          max_eval=0, teacher=None):
    """Evaluate multiple teacher-logit fusion weights in one validation pass."""
    if teacher is None:
        raise ValueError("A teacher model is required for fusion sweep evaluation.")
    model.eval()
    teacher.eval()
    metrics_by_weight = {weight: StreamSegMetrics(num_classes) for weight in fusion_weights}
    for metrics in metrics_by_weight.values():
        metrics.reset()

    for idx, batch in enumerate(loader):
        if max_eval and idx >= max_eval:
            break
        images_um = batch["image_um"].to(device, dtype=torch.float32)
        images_sam2 = batch["image_sam2"].to(device, dtype=torch.float32)
        labels = batch["label"].to(device, dtype=torch.long)
        logits, _, _, _, _ = forward_model(model, images_um, images_sam2, total_itrs, total_itrs)
        teacher_logits = teacher(images_um)
        if teacher_logits.shape[2:] != logits.shape[2:]:
            teacher_logits = F.interpolate(
                teacher_logits, size=logits.shape[2:], mode="bilinear", align_corners=False,
            )
        targets = labels.cpu().numpy()
        targets = np.where(targets < 0, 255, targets)
        for weight, metrics in metrics_by_weight.items():
            preds = ((1.0 - weight) * logits + weight * teacher_logits).argmax(dim=1).cpu().numpy()
            metrics.update(targets, preds)
    return {weight: metrics.get_results() for weight, metrics in metrics_by_weight.items()}


def split_trainable_params(model):
    sam_params = []
    mamba_params = []
    sam2_backbone_params = []

    mamba_keywords = (
        "umamba.", "umamba_decoder", "pam",
        "align_um", "residual_um", "cross_attn_u_from_s", "um_logit_scale",
    )
    sam_keywords = (
        "sam_proj", "sam_seg_head", "fusion", "fused_logit_scale",
        "sam_logit_scale", "align_sam", "residual_sam", "cross_attn_s_from_u",
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("sam2."):
            sam2_backbone_params.append(param)
        elif any(key in name for key in mamba_keywords):
            mamba_params.append(param)
        elif any(key in name for key in sam_keywords):
            sam_params.append(param)
        else:
            sam_params.append(param)

    return sam2_backbone_params, sam_params, mamba_params


def build_optimizers(target, opts, logger):
    sam2_backbone_params, sam_params, mamba_params = split_trainable_params(target)

    if opts.optimizer_mode == "legacy_adamw":
        legacy_groups = []
        if sam2_backbone_params:
            legacy_groups.append({"params": sam2_backbone_params, "lr": opts.lr_backbone})
        if mamba_params:
            legacy_groups.append({"params": mamba_params, "lr": opts.lr})
        if sam_params:
            legacy_groups.append({"params": sam_params, "lr": opts.lr * opts.head_lr_mult})
        optimizer = torch.optim.AdamW(legacy_groups, weight_decay=opts.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opts.total_itrs, eta_min=max(opts.lr * 0.01, 1e-7),
        )
        logger.info(
            "Param groups: sam2_backbone=%d, sam_other=%d, mamba=%d (legacy AdamW)",
            sum(p.numel() for p in sam2_backbone_params),
            sum(p.numel() for p in sam_params),
            sum(p.numel() for p in mamba_params),
        )
        return optimizer, scheduler, None, None

    sam_groups = []
    if sam2_backbone_params:
        sam_groups.append({"params": sam2_backbone_params, "lr": opts.lr_backbone})
    if sam_params:
        sam_groups.append({"params": sam_params, "lr": opts.lr})

    optimizer_sam = None
    scheduler_sam = None
    if sam_groups:
        optimizer_sam = torch.optim.AdamW(
            sam_groups, weight_decay=opts.weight_decay,
        )
        scheduler_sam = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_sam, T_max=opts.total_itrs, eta_min=max(opts.lr * 0.01, 1e-7),
        )

    optimizer_mamba = None
    scheduler_mamba = None
    if mamba_params:
        optimizer_mamba = torch.optim.SGD(
            mamba_params,
            lr=opts.lr_mamba,
            momentum=opts.mamba_momentum,
            weight_decay=opts.weight_decay,
            nesterov=True,
        )
        scheduler_mamba = PolyLR(optimizer_mamba, opts.total_itrs, power=0.9)

    logger.info(
        "Param groups: sam2_backbone=%d, sam_other=%d, mamba=%d",
        sum(p.numel() for p in sam2_backbone_params),
        sum(p.numel() for p in sam_params),
        sum(p.numel() for p in mamba_params),
    )
    return optimizer_sam, scheduler_sam, optimizer_mamba, scheduler_mamba


def zero_grad_all(*optimizers):
    for optimizer in optimizers:
        if optimizer is not None:
            optimizer.zero_grad()


def step_all(*optimizers):
    for optimizer in optimizers:
        if optimizer is not None:
            optimizer.step()


def scheduler_step_all(*schedulers):
    for scheduler in schedulers:
        if scheduler is not None:
            scheduler.step()


def first_lr(optimizer):
    if optimizer is None or not optimizer.param_groups:
        return 0.0
    return optimizer.param_groups[0]["lr"]


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/zlt/projects/data/voc2012")
    parser.add_argument("--train_split", type=str, default="train",
                        choices=["train", "train_aug", "trainval"])
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--total_itrs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="SAM-side/DFI/fusion learning rate (manuscript: 3e-5).")
    parser.add_argument("--lr_backbone", type=float, default=6e-6,
                        help="SAM2 image-encoder learning rate if --train_sam2 is used.")
    parser.add_argument("--lr_mamba", type=float, default=0.01,
                        help="U-Mamba/PAM learning rate for SGD+Nesterov.")
    parser.add_argument("--mamba_momentum", type=float, default=0.99)
    parser.add_argument("--optimizer_mode", choices=["manuscript", "legacy_adamw"], default="manuscript",
                        help="Use manuscript AdamW+SGD or the original checkpoint-compatible all-AdamW mode.")
    parser.add_argument("--head_lr_mult", type=float, default=1.0,
                        help="Head learning-rate multiplier used by --optimizer_mode legacy_adamw.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="balanced_ce_dice",
                        choices=["ce", "ce_dice", "balanced_ce_dice", "foreground_ce_dice"])
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=0.25)
    parser.add_argument("--bcc_weight", type=float, default=0.05)
    parser.add_argument("--bcc_warmup_itrs", type=int, default=0,
                        help="Keep BCC disabled for this many iterations, then ramp it up linearly.")
    parser.add_argument("--bcc_tau", type=float, default=0.9)
    parser.add_argument("--features", type=int, nargs="+",
                        default=[32, 64, 128, 256, 256, 256])
    parser.add_argument("--strides", type=int, nargs="+",
                        default=[1, 2, 2, 2, 2, 2])
    parser.add_argument("--n_blocks", type=int, nargs="+",
                        default=[2, 2, 2, 2, 2, 2])
    parser.add_argument("--dfi_stages", type=int, default=1,
                        help="Use low-resolution SAM2 FPN stages only; 1 is stable for VOC probes.")
    parser.add_argument("--dfi_align_channels", type=int, default=128)
    parser.add_argument("--dfi_num_heads", type=int, default=4)
    parser.add_argument("--pam_tau", type=float, default=0.5)
    parser.add_argument("--freeze_sam2_all", action="store_true", default=True)
    parser.add_argument("--train_sam2", dest="freeze_sam2_all", action="store_false")
    parser.add_argument("--sam2_train_last_blocks", type=int, default=0,
                        help="Fine-tune only the last N SAM2 Hiera blocks plus its FPN neck.")
    parser.add_argument("--sam2_repo", type=str, default="/home/zlt/projects/sam2")
    parser.add_argument("--sam2_config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2_ckpt", type=str, default="checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--deeplab_repo", type=str, default="/home/zlt/projects/DeepLabV3Plus-Pytorch")
    parser.add_argument("--teacher_model", type=str, default="deeplabv3plus_mobilenet")
    parser.add_argument("--teacher_output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--teacher_ckpt", type=str,
                        default="/home/zlt/projects/DeepLabV3Plus-Pytorch/checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth")
    parser.add_argument("--teacher_weight", type=float, default=0.3)
    parser.add_argument("--teacher_temperature", type=float, default=2.0)
    parser.add_argument("--eval_teacher_fusion_weight", type=float, default=0.0,
                        help="Validation/eval logit fusion weight for the fixed DeepLab teacher. 0 disables it.")
    parser.add_argument("--eval_teacher_fusion_sweep", type=float, nargs="+", default=None,
                        help="In --eval_only mode, evaluate these fusion weights in one validation pass.")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--continue_training", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="checkpoints/ddktn_voc_miou_probe")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=250)
    parser.add_argument("--max_val_eval", type=int, default=0)
    parser.add_argument("--stop_after_val", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--gpu_id", type=str, default="0")
    return parser


def main():
    opts = get_argparser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    logger = setup_logger(opts.save_dir)
    logger.info("=" * 60)
    logger.info("Device: %s", device)
    logger.info("Config: %s", vars(opts))

    train_loader = create_voc_dataloader(
        opts.data_root,
        batch_size=opts.batch_size,
        split=opts.train_split,
        target_size=(opts.crop_size, opts.crop_size),
        num_classes=opts.num_classes,
        num_workers=opts.num_workers,
        augment=True,
    )
    val_loader = create_official_voc513_val_dataloader(
        opts.data_root,
        batch_size=opts.val_batch_size,
        target_size=(opts.crop_size, opts.crop_size),
        num_classes=opts.num_classes,
        num_workers=opts.num_workers,
    )
    logger.info("Dataset: Train split=%s, Train=%d, Val=%d (Resize(%d)+CenterCrop(%d))", opts.train_split, len(train_loader.dataset), len(val_loader.dataset), opts.crop_size, opts.crop_size)

    model = build_model(opts, device)
    teacher = build_teacher(opts, device, logger)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info("Using %d GPUs via DataParallel", torch.cuda.device_count())

    target = model.module if hasattr(model, "module") else model
    optimizer_sam, scheduler_sam, optimizer_mamba, scheduler_mamba = build_optimizers(
        target, opts, logger
    )
    criterion = SegmentationLoss(opts.loss, ignore_index=-1, dice_weight=opts.dice_weight)
    metrics = StreamSegMetrics(opts.num_classes)
    bcc_module = BCCModule(tau_c=opts.bcc_tau, num_classes=opts.num_classes).to(device)

    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt and os.path.isfile(opts.ckpt):
        ckpt = torch.load(opts.ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        target.load_state_dict(state, strict=False)
        if opts.continue_training:
            if optimizer_sam is not None and "optimizer_sam_state" in ckpt:
                optimizer_sam.load_state_dict(ckpt["optimizer_sam_state"])
            if optimizer_mamba is not None and "optimizer_mamba_state" in ckpt:
                optimizer_mamba.load_state_dict(ckpt["optimizer_mamba_state"])
            if scheduler_sam is not None and "scheduler_sam_state" in ckpt:
                scheduler_sam.load_state_dict(ckpt["scheduler_sam_state"])
            if scheduler_mamba is not None and "scheduler_mamba_state" in ckpt:
                scheduler_mamba.load_state_dict(ckpt["scheduler_mamba_state"])
                scheduler_mamba.max_iters = opts.total_itrs
            cur_itrs = ckpt.get("cur_itrs", 0)
            best_score = ckpt.get("best_score", 0.0)
        logger.info("Loaded checkpoint: %s", opts.ckpt)

    def save_ckpt(path):
        target_model = model.module if hasattr(model, "module") else model
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": target_model.state_dict(),
            "optimizer_sam_state": optimizer_sam.state_dict() if optimizer_sam is not None else None,
            "optimizer_mamba_state": optimizer_mamba.state_dict() if optimizer_mamba is not None else None,
            "scheduler_sam_state": scheduler_sam.state_dict() if scheduler_sam is not None else None,
            "scheduler_mamba_state": scheduler_mamba.state_dict() if scheduler_mamba is not None else None,
            "best_score": best_score,
            "opts": vars(opts),
        }, path)
        logger.info("Saved: %s", path)

    logger.info(
        "Params: %.2fM total / %.2fM trainable",
        sum(p.numel() for p in target.parameters()) / 1e6,
        sum(p.numel() for p in target.parameters() if p.requires_grad) / 1e6,
    )
    if opts.optimizer_mode == "legacy_adamw":
        logger.info(
            "LR groups: SAM2=%.6g, U-Mamba=%.6g, SAM/DFI/head=%.6g (AdamW)",
            opts.lr_backbone, opts.lr, opts.lr * opts.head_lr_mult,
        )
    else:
        logger.info(
            "LR groups: sam=%.6g, sam_backbone=%.6g, mamba=%.6g (SGD Nesterov %.2f)",
            opts.lr, opts.lr_backbone, opts.lr_mamba, opts.mamba_momentum,
        )
    if opts.eval_teacher_fusion_weight > 0:
        logger.info("Eval teacher logit fusion weight: %.3f", opts.eval_teacher_fusion_weight)

    if opts.eval_only:
        if opts.eval_teacher_fusion_sweep:
            sweep_scores = validate_fusion_sweep(
                model, val_loader, device, max(opts.total_itrs, 1),
                opts.eval_teacher_fusion_sweep, opts.num_classes,
                max_eval=opts.max_val_eval, teacher=teacher,
            )
            with open(os.path.join(opts.save_dir, "val.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                for weight, val_score in sweep_scores.items():
                    logger.info("Eval fusion weight %.3f:%s", weight, StreamSegMetrics.to_str(val_score))
                    writer.writerow([
                        cur_itrs, cur_epochs, f"{weight:.3f}", f"{val_score['Mean IoU']:.6f}",
                        f"{val_score['Overall Acc']:.6f}", f"{val_score['Mean Acc']:.6f}",
                    ])
            return
        val_score = validate(
            model, val_loader, device, metrics, max(opts.total_itrs, 1),
            max_eval=opts.max_val_eval,
            teacher=teacher,
            teacher_fusion_weight=opts.eval_teacher_fusion_weight,
        )
        logger.info("Eval only:%s", StreamSegMetrics.to_str(val_score))
        with open(os.path.join(opts.save_dir, "val.csv"), "a", newline="") as f:
            csv.writer(f).writerow([
                cur_itrs, cur_epochs, f"{val_score['Mean IoU']:.6f}",
                f"{val_score['Overall Acc']:.6f}", f"{val_score['Mean Acc']:.6f}",
            ])
        return

    interval_loss = 0.0
    interval_main = 0.0
    t_start = time.time()
    while cur_itrs < opts.total_itrs:
        model.train()
        cur_epochs += 1
        for batch in train_loader:
            cur_itrs += 1
            images_um = batch["image_um"].to(device, dtype=torch.float32)
            images_sam2 = batch["image_sam2"].to(device, dtype=torch.float32)
            labels = batch["label"].to(device, dtype=torch.long)

            zero_grad_all(optimizer_sam, optimizer_mamba)
            logits, logits_sam, logits_um, prob_um, bcc_info = forward_model(
                model, images_um, images_sam2, cur_itrs, opts.total_itrs
            )
            main_loss = criterion(logits, labels)
            loss = main_loss
            if opts.aux_weight > 0:
                loss = loss + opts.aux_weight * criterion(logits_sam, labels)
                loss = loss + opts.aux_weight * criterion(logits_um, labels)
            if opts.bcc_weight > 0 and cur_itrs > opts.bcc_warmup_itrs:
                p_sam = F.softmax(logits_sam.detach(), dim=1)
                p_um = F.softmax(logits_um.detach(), dim=1)
                bcc_loss, _, _ = bcc_module(
                    p_sam, p_um,
                    f_sam=bcc_info["f_sam"],
                    f_um=bcc_info["f_um"],
                )
                if opts.bcc_warmup_itrs > 0:
                    progress = min((cur_itrs - opts.bcc_warmup_itrs) / opts.bcc_warmup_itrs, 1.0)
                else:
                    progress = 1.0
                loss = loss + (opts.bcc_weight * progress) * bcc_loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images_um)
                kd_loss = distillation_loss(
                    logits, teacher_logits, labels, opts.teacher_temperature,
                )
                loss = loss + opts.teacher_weight * kd_loss
            loss.backward()
            nn.utils.clip_grad_norm_(target.parameters(), max_norm=1.0)
            step_all(optimizer_sam, optimizer_mamba)
            scheduler_step_all(scheduler_sam, scheduler_mamba)

            interval_loss += float(loss.detach().cpu())
            interval_main += float(main_loss.detach().cpu())
            if cur_itrs % opts.print_interval == 0:
                avg_loss = interval_loss / opts.print_interval
                avg_main = interval_main / opts.print_interval
                interval_loss = 0.0
                interval_main = 0.0
                lr_sam = first_lr(optimizer_sam)
                lr_mamba = first_lr(optimizer_mamba)
                speed = cur_itrs / max(time.time() - t_start, 1e-6)
                logger.info(
                    "Epoch %d, Itrs %d/%d, Loss=%.4f, Main=%.4f, LR_sam=%.6g, LR_mamba=%.6g, Speed=%.2f it/s",
                    cur_epochs, cur_itrs, opts.total_itrs, avg_loss, avg_main, lr_sam, lr_mamba, speed,
                )
                with open(os.path.join(opts.save_dir, "loss.csv"), "a", newline="") as f:
                    csv.writer(f).writerow([
                        cur_itrs, cur_epochs, f"{avg_loss:.6f}",
                        f"{avg_main:.6f}", f"{lr_sam:.8f}", f"{lr_mamba:.8f}",
                    ])

            if cur_itrs % opts.val_interval == 0:
                save_ckpt(os.path.join(opts.save_dir, "latest.pth"))
                val_score = validate(
                    model, val_loader, device, metrics, opts.total_itrs,
                    max_eval=opts.max_val_eval,
                    teacher=teacher,
                    teacher_fusion_weight=opts.eval_teacher_fusion_weight,
                )
                logger.info(StreamSegMetrics.to_str(val_score))
                with open(os.path.join(opts.save_dir, "val.csv"), "a", newline="") as f:
                    csv.writer(f).writerow([
                        cur_itrs, cur_epochs, f"{val_score['Mean IoU']:.6f}",
                        f"{val_score['Overall Acc']:.6f}", f"{val_score['Mean Acc']:.6f}",
                    ])
                if val_score["Mean IoU"] > best_score:
                    best_score = val_score["Mean IoU"]
                    save_ckpt(os.path.join(opts.save_dir, "best.pth"))
                if opts.stop_after_val:
                    save_ckpt(os.path.join(opts.save_dir, "final.pth"))
                    return

            if cur_itrs >= opts.total_itrs:
                break

    val_score = validate(
        model, val_loader, device, metrics, opts.total_itrs,
        max_eval=opts.max_val_eval,
        teacher=teacher,
        teacher_fusion_weight=opts.eval_teacher_fusion_weight,
    )
    logger.info("Final validation:%s", StreamSegMetrics.to_str(val_score))
    save_ckpt(os.path.join(opts.save_dir, "final.pth"))


if __name__ == "__main__":
    main()

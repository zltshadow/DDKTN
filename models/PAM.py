"""
Prompt Adaptation Module (PAM)
Automatically generates box and point prompts from U-Mamba predictions
for SAM2, eliminating the need for manual prompting.

Implements symmetry-induced spatial decoupling for bilateral orbital anatomy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


class PAM(nn.Module):
    """Prompt Adaptation Module.

    Generates structured prompts (bounding boxes + positive/negative points)
    from U-Mamba's coarse predictions, which are then fed to SAM2's prompt
    encoder for refined segmentation.

    Key steps:
    1. Binarize U-Mamba probability maps with high threshold (tau=0.9)
    2. Symmetry-induced spatial decoupling (left/right subspaces)
    3. Compute bounding boxes from spatial extrema
    4. Generate positive prompts from geometric centroids
    5. Hard negative mining from morphological boundary regions
    """
    def __init__(self, tau=0.9, num_classes=4, bilateral=True):
        """
        Args:
            tau: binarization threshold for probability maps
            num_classes: number of segmentation classes
            bilateral: whether to use bilateral symmetry decoupling
        """
        super().__init__()
        self.tau = tau
        self.num_classes = num_classes
        self.bilateral = bilateral

    def binarize_prob(self, prob_map):
        """Binarize probability maps using threshold tau.

        Args:
            prob_map: (B, C, H, W) - probability map from U-Mamba
        Returns:
            binary_masks: (B, C, H, W) - binarized masks
        """
        return (prob_map > self.tau).float()

    def split_bilateral(self, masks):
        """Split masks into left and right subspaces for bilateral anatomy.

        Args:
            masks: (B, C, H, W) - binary masks
        Returns:
            left_masks: (B, C, H, W//2) - left half
            right_masks: (B, C, H, W//2) - right half
        """
        B, C, H, W = masks.shape
        mid = W // 2
        left_masks = masks[:, :, :, :mid]
        right_masks = masks[:, :, :, mid:]
        return left_masks, right_masks

    def compute_bounding_box(self, mask):
        """Compute bounding box from binary mask.

        Args:
            mask: (H, W) - binary mask for a single class/side
        Returns:
            bbox: [x_min, y_min, x_max, y_max] or None if mask is empty
        """
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.shape[0] == 0:
            return None
        y_min, x_min = coords.min(dim=0)[0]
        y_max, x_max = coords.max(dim=0)[0]
        return [x_min.item(), y_min.item(), x_max.item(), y_max.item()]

    def compute_centroid(self, mask):
        """Compute geometric centroid using first-order spatial moments.

        Args:
            mask: (H, W) - binary mask
        Returns:
            centroid: (x, y) or None if mask is empty
        """
        if mask.sum() == 0:
            return None
        # First-order spatial moments
        y_coords, x_coords = torch.where(mask > 0)
        cx = x_coords.float().mean()
        cy = y_coords.float().mean()
        return (cx.item(), cy.item())

    def mine_hard_negatives(self, mask, num_points=1, dilate_size=5):
        """Mine hard negative points from boundary region.

        Boundary = Dilate(mask) - mask

        Args:
            mask: (H, W) - binary mask
            num_points: number of negative points to sample
            dilate_size: dilation kernel size
        Returns:
            neg_points: list of (x, y) tuples
        """
        mask_np = mask.cpu().numpy().astype(np.uint8)
        if mask_np.sum() == 0:
            return []

        # Morphological dilation
        struct = ndimage.generate_binary_structure(2, 1)
        dilated = ndimage.binary_dilation(mask_np, structure=struct, iterations=dilate_size // 2)
        boundary = dilated.astype(np.float32) - mask_np

        boundary_coords = np.argwhere(boundary > 0)
        if len(boundary_coords) == 0:
            return []

        # Random sampling from boundary
        indices = np.random.choice(len(boundary_coords), size=min(num_points, len(boundary_coords)), replace=False)
        neg_points = [(int(boundary_coords[i, 1]), int(boundary_coords[i, 0])) for i in indices]
        return neg_points

    def generate_prompts(self, prob_map):
        """Generate box and point prompts from U-Mamba probability map.

        Args:
            prob_map: (B, C, H, W) - U-Mamba probability map
        Returns:
            box_prompts: list of lists of [x_min, y_min, x_max, y_max] per class per sample
            pos_point_prompts: list of lists of (x, y) per class per sample
            neg_point_prompts: list of lists of (x, y) per class per sample
        """
        B, C, H, W = prob_map.shape
        binary_masks = self.binarize_prob(prob_map)

        all_boxes = []
        all_pos_points = []
        all_neg_points = []

        for b in range(B):
            sample_boxes = []
            sample_pos = []
            sample_neg = []

            for c in range(1, C):  # Skip background class 0
                mask = binary_masks[b, c]  # (H, W)

                if self.bilateral:
                    # Split into left and right subspaces
                    mid = W // 2
                    left_mask = mask[:, :mid]
                    right_mask = mask[:, mid:]

                    for side_mask, offset in [(left_mask, 0), (right_mask, mid)]:
                        bbox = self.compute_bounding_box(side_mask)
                        if bbox is not None:
                            # Adjust x coordinates for right side
                            bbox[0] += offset
                            bbox[2] += offset
                            sample_boxes.append(bbox)

                            centroid = self.compute_centroid(side_mask)
                            if centroid is not None:
                                sample_pos.append((centroid[0] + offset, centroid[1]))

                            neg_pts = self.mine_hard_negatives(side_mask, num_points=1)
                            for pt in neg_pts:
                                sample_neg.append((pt[0] + offset, pt[1]))
                else:
                    bbox = self.compute_bounding_box(mask)
                    if bbox is not None:
                        sample_boxes.append(bbox)

                    centroid = self.compute_centroid(mask)
                    if centroid is not None:
                        sample_pos.append(centroid)

                    neg_pts = self.mine_hard_negatives(mask, num_points=1)
                    sample_neg.extend(neg_pts)

            all_boxes.append(sample_boxes)
            all_pos_points.append(sample_pos)
            all_neg_points.append(sample_neg)

        return all_boxes, all_pos_points, all_neg_points

    def prompts_to_tensor(self, box_prompts, pos_points, neg_points, image_size=1024):
        """Convert prompts to tensor format compatible with SAM2 prompt encoder.

        Args:
            box_prompts: list of lists of [x_min, y_min, x_max, y_max]
            pos_points: list of lists of (x, y)
            neg_points: list of lists of (x, y)
            image_size: target image size for coordinate normalization
        Returns:
            point_coords: (B, N, 2) - point coordinates in SAM2 input space
            point_labels: (B, N) - point labels (1=pos, 0=neg, 2=box_tl, 3=box_br)
            box_coords: (B, 4) - box coordinates in SAM2 input space
        """
        B = len(box_prompts)
        device = 'cpu'  # Will be moved to correct device later

        all_coords = []
        all_labels = []
        all_boxes = []

        for b in range(B):
            coords = []
            labels = []

            # Add positive points (label=1)
            for pt in pos_points[b]:
                coords.append(list(pt))
                labels.append(1)

            # Add negative points (label=0)
            for pt in neg_points[b]:
                coords.append(list(pt))
                labels.append(0)

            # Convert to tensor
            if len(coords) > 0:
                coords_tensor = torch.tensor(coords, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.int32)
            else:
                coords_tensor = torch.zeros((1, 2), dtype=torch.float32)
                labels_tensor = torch.tensor([-1], dtype=torch.int32)  # Padding

            all_coords.append(coords_tensor)
            all_labels.append(labels_tensor)

            # Box prompts - use the first box if available
            if len(box_prompts[b]) > 0:
                box = box_prompts[b][0]  # Use first box
                all_boxes.append(torch.tensor(box, dtype=torch.float32))
            else:
                all_boxes.append(torch.zeros(4, dtype=torch.float32))

        # Pad to same length
        max_len = max(c.shape[0] for c in all_coords)
        padded_coords = torch.zeros(B, max_len, 2, dtype=torch.float32)
        padded_labels = torch.full((B, max_len), -1, dtype=torch.int32)

        for b in range(B):
            L = all_coords[b].shape[0]
            padded_coords[b, :L] = all_coords[b]
            padded_labels[b, :L] = all_labels[b]

        box_tensor = torch.stack(all_boxes, dim=0)

        return padded_coords, padded_labels, box_tensor

    def forward(self, prob_map_um, image_size=1024):
        """Generate prompts from U-Mamba probability map.

        Args:
            prob_map_um: (B, C, H, W) - U-Mamba probability map
            image_size: target image size for SAM2 input
        Returns:
            point_coords: (B, N, 2) - point coordinates
            point_labels: (B, N) - point labels
            box_coords: (B, 4) - box coordinates
        """
        boxes, pos_pts, neg_pts = self.generate_prompts(prob_map_um)

        # Scale coordinates to SAM2 input space
        H, W = prob_map_um.shape[2], prob_map_um.shape[3]
        scale_x = image_size / W
        scale_y = image_size / H

        # Scale all coordinates
        scaled_pos = []
        scaled_neg = []
        scaled_boxes = []

        for b in range(len(boxes)):
            sp = [(x * scale_x, y * scale_y) for x, y in pos_pts[b]]
            sn = [(x * scale_x, y * scale_y) for x, y in neg_pts[b]]
            sb = []
            for box in boxes[b]:
                sb.append([
                    box[0] * scale_x, box[1] * scale_y,
                    box[2] * scale_x, box[3] * scale_y
                ])
            scaled_pos.append(sp)
            scaled_neg.append(sn)
            scaled_boxes.append(sb)

        point_coords, point_labels, box_coords = self.prompts_to_tensor(
            scaled_boxes, scaled_pos, scaled_neg, image_size
        )

        return point_coords, point_labels, box_coords

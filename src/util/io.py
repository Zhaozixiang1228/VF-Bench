# Last modified: 2025-04-17

import csv
import torch
import numpy as np
import einops
from PIL import Image


def read_csv(filename, delimiter=","):
    with open(filename, "r", newline="") as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        header = next(csv_reader)
        content = [row for row in csv_reader if row]
    return header, content


def pred_2_8bit(pred: torch.Tensor, src1: torch.Tensor, src2: torch.Tensor) -> np.ndarray:
    """
    pred: [B, 3, 3, H, W]
    src1: [B, 5, 3, H, W]
    src2: [B, 5, 3, H, W]
    """
    vis = pred[:, 1, :, :, :].detach().cpu().clone().numpy().squeeze()  # [3, H, W]
    assert vis.ndim == 3, "pred should be a 3D tensor"

    src1=src1[:, 2, :, :, :].detach().cpu().clone().numpy().squeeze()
    src2=src2[:, 2, :, :, :].detach().cpu().clone().numpy().squeeze()

    vis = einops.rearrange(vis, "c h w -> h w c")
    src1 = einops.rearrange(src1, "c h w -> h w c")
    src2 = einops.rearrange(src2, "c h w -> h w c")
    vis =np.concatenate([src1,src2,vis],1)
    vis = (vis * 255).astype(np.uint8)
    return vis


def save_image(image: np.ndarray, filename: str):
    if image.ndim == 2:  # Grayscale
        image = np.expand_dims(image, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:  # Single channel
        image = np.squeeze(image, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 3:  # RGB
        pass
    else:
        raise ValueError("Image must be either grayscale or RGB format.")

    im = Image.fromarray(image)
    im.save(filename)

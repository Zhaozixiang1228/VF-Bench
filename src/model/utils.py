"""
Util functions for network construction
"""
import os
import importlib
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def coords_grid(b, h, w, device):
    coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(b, 1, 1, 1)

def backward_warp(img, flow, pad='zeros'):
    b, c, h, w = img.shape
    grid = coords_grid(b, h, w, device=img.device)
    grid = grid + flow
    xgrid, ygrid = grid.split([1,1], dim=1)
    xgrid = 2*xgrid/(w-1) - 1
    ygrid = 2*ygrid/(h-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=1)
    warped_img = F.grid_sample(input=img, grid=grid.permute(0,2,3,1), mode='bilinear',  padding_mode='zeros') 
    return warped_img


def load_args_from_json(json_path):
    """
    Load parameters from a JSON configuration file and return an object with attribute access.
    """
    with open(json_path, 'r') as f:
        config_dict = json.load(f)

    # Convert dictionary to an object that allows dot notation access
    class Args: pass
    args = Args()
    args.__dict__.update(config_dict)
    
    return args

def flow_warp(input_img, flow): 
    """
    Warp input_img into the target frame coordinate space.
    
    input_img: [B, C, H, W]
    flow:      [B, 2, H, W], optical flow from input_img to the target frame
    return:    [B, C, H, W] warped image

    Usage:
    I1_warped = flow_warp(I1, flow_I2_to_I1)
    I3_warped = flow_warp(I3, flow_I2_to_I3)
    """
    B, C, H, W = input_img.shape
    # Normalize coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=input_img.device),
        torch.linspace(-1, 1, W, device=input_img.device),
        indexing='ij'
    )
    base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).expand(B, H, W, 2)  # [B, H, W, 2]

    # Normalize flow to [-1, 1] range
    norm_flow = torch.zeros_like(base_grid)
    norm_flow[..., 0] = flow[:, 0, :, :] * (2.0 / (W - 1))
    norm_flow[..., 1] = flow[:, 1, :, :] * (2.0 / (H - 1))

    # Compute backward sampling coordinates
    sample_grid = base_grid + norm_flow

    # Bilinear sampling
    warped = F.grid_sample(input_img, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped
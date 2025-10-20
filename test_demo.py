# -*- coding: utf-8 -*-
# Last modified: 2025-10-20
# Author: Zixiang Zhao

import os
# Set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import logging
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

from src.dataset.base_two_modal_dataset import DatasetMode
from src.dataset import get_multi_frame_dataset
from src.model.net import VideoFusion
from src.util.io import pred_2_8bit, save_image
from src.util.logging_util import setup_logging
from src.util.video_util import generate_video_from_image_paths


import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Model testing script")
    parser.add_argument("--exp_path", type=str, default='UniVF-MEF',
                        help="Experiment directory")
    parser.add_argument(
        "--task_name",
        type=str,
        default="MEF",
        help="Task name (IVF, MEF, MFF, MVF)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="YouTube-demo",
        help="Dataset name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size during testing (evaluation metrics usually require 1)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run testing on (e.g., 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frame rate for generated videos",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=50000,
        help="Bitrate for generated videos",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Logging setup ---
    setup_logging(
        os.path.join("output_demo", args.task_name, args.dataset_name),
        file_log_level="INFO",
        console_log_level=args.log_level.upper(),
    )
    logging.info(f"Testing script started with args: {args}")

    # --- Device and configuration loading ---
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    if args.task_name == "IVF":
        args.config = "config/train/ivf-train.yaml"
        if args.dataset_name == "VTMOT":
            args.data_cfg = "config/dataset/IVF/VTMOT/vtmot_5-frame.yaml"
        elif args.dataset_name == "VTMOT-demo":
            args.data_cfg = "config/dataset/demo/vtmot_demo_5-frame.yaml"
        else:
            raise ValueError(f"Unsupported dataset_name for IVF: {args.dataset_name}")
    elif args.task_name == "MEF":
        args.config = "config/train/mef-train.yaml"
        if args.dataset_name == "YouTube":
            args.data_cfg = "config/dataset/MEF/YouTubeHDR/youtube_5-frame.yaml"
        elif args.dataset_name == "YouTube-540p":
            args.data_cfg = "config/dataset/MEF/YouTubeHDR/youtube_5-frame_540p.yaml"
        elif args.dataset_name == "YouTube-demo":
            args.data_cfg = "config/dataset/demo/youtube_demo_5-frame.yaml"
        else:
            raise ValueError(f"Unsupported dataset_name for MEF: {args.dataset_name}")
    elif args.task_name == "MFF":
        args.config = "config/train/mff-train.yaml"
        if args.dataset_name == "DAVIS":
            args.data_cfg = "config/dataset/MFF/DAVIS/davis_5-frame.yaml"
        elif args.dataset_name == "DAVIS-480p":
            args.data_cfg = "config/dataset/MFF/DAVIS/davis_5-frame_480p.yaml"
        elif args.dataset_name == "DAVIS-demo":
            args.data_cfg = "config/dataset/demo/davis_demo_5-frame.yaml"
        else:
            raise ValueError(f"Unsupported dataset_name for MFF: {args.dataset_name}")
    elif args.task_name == "MVF":
        args.config = "config/train/mvf-train.yaml"
        if args.dataset_name == "Harvard":
            args.data_cfg = "config/dataset/MVF/Harvard/harvard_5-frame.yaml"
        elif args.dataset_name == "Harvard-demo":
            args.data_cfg = "config/dataset/demo/harvard_demo_5-frame.yaml"
        else:
            raise ValueError(f"Unsupported dataset_name for MVF: {args.dataset_name}")
    else:
        raise ValueError(f"Unsupported task_name: {args.task_name}")

    cfg = OmegaConf.load(args.config)

    # --- Model weights ---
    model_path = os.path.join(
        "output", args.exp_path, "checkpoint", "latest", "model.pth"
    )

    if not os.path.exists(model_path):
        logging.error(f"Model checkpoint not found: {model_path}")
        sys.exit(1)

    # --- Model loading ---
    logging.info("Initializing model...")
    model = (
        VideoFusion(model_config = cfg).to(device).eval()
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f"Model weights successfully loaded from {model_path}")

    # --- Dataset loading ---
    logging.info(f"Loading test dataset: {args.data_cfg}")
    data_cfg = OmegaConf.load(args.data_cfg)
    dataset = get_multi_frame_dataset(
        data_cfg, base_data_dir="data", mode=DatasetMode.TEST
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    logging.info(f"Dataset '{args.dataset_name}' loaded with {len(dataset)} samples.")

    # --- Output directory ---
    vis_dir = os.path.join(
        "output_demo", args.task_name, args.dataset_name
    )
    os.makedirs(vis_dir, exist_ok=True)
    
    # --- Inference loop ---
    logging.info("Starting inference loop...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Testing {args.dataset_name}")):
            # --- Model inference ---
            if args.task_name == "IVF":
                I_1 = batch["ir"].to(device)   # B, T, C, H, W
                I_2 = batch["rgb"].to(device)  # B, T, C, H, W
            elif args.task_name == "MEF":
                I_1 = batch["under"].to(device)
                I_2 = batch["over"].to(device)
            elif args.task_name == "MFF":
                I_1 = batch["far"].to(device)
                I_2 = batch["near"].to(device)
            elif args.task_name == "MVF":
                I_1 = batch["mri"].to(device)
                I_2 = batch["other"].to(device)
            else:
                raise ValueError(f"Unsupported task_name: {args.task_name}")

            fusion_pred, _ = model(I_1, I_2)
            if torch.isnan(fusion_pred).any():
                logging.warning("fusion_pred contains NaN values.")
            
            # --- Save visualization ---
            vis_8bit = pred_2_8bit(fusion_pred, I_1, I_2)  # Convert to 8-bit image
            middle_idx = I_1.shape[1] // 2  # Take the middle frame
            if args.task_name == "IVF":
                ir_path = batch["data_path_ls_dict"]["ir"][middle_idx][0]
            elif args.task_name == "MEF":
                ir_path = batch["data_path_ls_dict"]["under"][middle_idx][0]
            elif args.task_name == "MFF":
                ir_path = batch["data_path_ls_dict"]["far"][middle_idx][0]
            elif args.task_name == "MVF":
                ir_path = batch["data_path_ls_dict"]["mri"][middle_idx][0]
            else:
                raise ValueError(f"Unsupported task_name: {args.task_name}")

            dir_name = os.path.dirname(ir_path).split("/")[0]  

            file_name = os.path.basename(ir_path)  # '000312.jpg'
            file_name = os.path.splitext(file_name)[0] + ".png"  # Change to '000312.png'

            save_path = os.path.join(vis_dir, dir_name)
            os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
            save_image(vis_8bit, os.path.join(save_path, file_name))

    for dir_name in os.listdir(vis_dir):
        dir_path = os.path.join(vis_dir, dir_name)
        if os.path.isdir(dir_path):
            logging.info("Merging images into video...")
            filenames = glob.glob(dir_path + "/*", recursive=False)
            filenames = sorted(filenames)
            logging.info(f"Found {len(filenames)} images")
            output_video_path = os.path.join(
                vis_dir, f"{dir_name}.mp4"
            )
            # Generate video
            generate_video_from_image_paths(
                image_paths=filenames,
                output_path=output_video_path,
                fps=args.fps,
                bitrate=args.bitrate,
                verbose=True,
            )

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
# -*- coding: utf-8 -*-
# Last modified: 2025-10-20
# Author: Zixiang Zhao

import os
import sys
import argparse
import logging
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import inspect
import csv
import pickle

# --- Import custom modules ---
from src.dataset.base_two_modal_dataset import DatasetMode
from src.dataset import get_multi_frame_dataset
from src.model.net import VideoFusion
from src.util import metric
from src.util.metric import MetricTracker, compute_metrics
from src.util.io import pred_2_8bit, save_image
from src.util.logging_util import eval_dic_to_text, setup_logging

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Model evaluation script")
    parser.add_argument("--exp_path", type=str, required=True, help="Experiment directory")
    parser.add_argument("--ckpt_path", type=str, help="Model checkpoint name")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name (IVF, MEF, MFF, MVF)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--base_data_dir", type=str, default="data", help="Base directory of the dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size during testing (usually 1 for evaluation metrics)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for testing (e.g., 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--no_save_vis",
        action="store_true",
        help="Do not save visualization results (by default, they are saved)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.exp_path:
        raise ValueError("Missing exp_path")

    ckpt_name = args.ckpt_path if args.ckpt_path else "latest"

    # --- Logging setup ---
    setup_logging(
        os.path.join("output", args.exp_path, "test_results", ckpt_name),
        file_log_level="INFO",
        console_log_level=args.log_level.upper(),
    )
    logging.info(f"Evaluation script started with args: {args}")

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
        "output", args.exp_path, "checkpoint", ckpt_name, "model.pth"
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
        data_cfg, base_data_dir=args.base_data_dir, mode=DatasetMode.TEST
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    dataset_name = getattr(
        dataset, "disp_name", os.path.splitext(os.path.basename(args.data_cfg))[0]
    )
    logging.info(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples.")

    # --- Metrics setup ---
    metric_funcs = []
    metric_tracker = None
    metric_names = cfg.eval.eval_metrics
    metric_funcs = [getattr(metric, name) for name in metric_names]
    metric_tracker = MetricTracker(*metric_names)
    logging.info(f"Using metrics: {metric_names}")
    data_dict = {"UniVF": {}}

    # --- Output directories ---
    vis_dir = os.path.join(
        "output", args.exp_path, "test_results", ckpt_name, "eval_visual", dataset_name
    )
    eval_dir = os.path.join("output", args.exp_path, "test_results", ckpt_name)
    if not args.no_save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # --- Inference loop ---
    logging.info("Starting inference loop...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Testing {dataset_name}")):
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

            # --- Save visualization results ---
            if not args.no_save_vis:
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
                # e.g., 'wurenji-0305-03/infrared' → 'wurenji-0305-03'

                file_name = os.path.basename(ir_path)  # '000312.jpg'
                file_name = os.path.splitext(file_name)[0] + ".png"  # → '000312.png'

                save_path = os.path.join(vis_dir, dir_name)
                os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
                save_image(vis_8bit, os.path.join(save_path, file_name))

            # --- Compute evaluation metrics ---
            if metric_tracker and metric_funcs:
                sample_metrics = compute_metrics(metric_funcs, fusion_pred, I_1, I_2)
                for k, v in sample_metrics.items():
                    metric_tracker.update(k, v)
                # --- Save per-frame metrics into data_dict ---
                if dir_name not in data_dict["UniVF"]:
                    data_dict["UniVF"][dir_name] = {k: [] for k in sample_metrics.keys()}
                for k, v in sample_metrics.items():
                    data_dict["UniVF"][dir_name][k].append(v)
            torch.cuda.empty_cache()  # Clear GPU cache

    # --- Record and save final metrics ---
    final_metrics = metric_tracker.result()
    logging.info(f"Final averaged metrics for dataset '{dataset_name}':")
    for key, value in final_metrics.items():
        logging.info(f"  {key}: {value}")

    # --- Save metrics to txt and csv files ---
    eval_base_name = os.path.join(eval_dir, f"eval-{dataset_name}")
    eval_text = eval_dic_to_text(
        final_metrics,
        f"Dataset: {dataset_name}\nCheckpoint: {model_path}",
    )

    with open(f"{eval_base_name}.txt", "w") as f:
        f.write(eval_text)

    # Save as vertical CSV
    csv_path = f"{eval_base_name}.csv"
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write metric names (header)
        writer.writerow(final_metrics.keys())
        # Write corresponding values
        writer.writerow(final_metrics.values())

    # --- Save per-frame metrics as pkl ---
    pkl_save_path = os.path.join(eval_dir, f"UniVF_metrics_{dataset_name}.pkl")
    with open(pkl_save_path, "wb") as f:
        pickle.dump(data_dict, f)

    logging.info(f"Per-frame metrics saved to: {pkl_save_path}")
    logging.info(f"Evaluation results saved to: {eval_base_name}.txt & {eval_base_name}.csv")


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
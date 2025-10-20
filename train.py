# -*- coding: utf-8 -*-
# Last modified: 2025-10-20
# Author: Zixiang Zhao

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import List
import pytz  # timezone
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator  # Multi-GPU training and mixed precision
from src.dataset import get_multi_frame_dataset, BaseTwoModalDataset, DatasetMode
from src.util.config_util import recursive_load_config
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
    create_code_snapshot,
)
from src.model.net import VideoFusion
import warnings
warnings.filterwarnings("ignore")

if "__main__" == __name__:

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--task_name", type=str, default="IVF", help="name of fusion task: MEF, MFF, IVF, MVF"
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="directory to save checkpoints"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "bf16", "fp16"]
    )
    parser.add_argument("--no_wandb", action="store_false", help="run without wandb")
    parser.add_argument(
        "--base_data_dir", type=str, default="./data", help="directory of training data"
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_false",
        help="Add datetime to the output folder name",
    )
    parser.add_argument(
        "--split_batch",
        action="store_true",
        help="Accelerator split batch",
    )

    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ["BASE_DATA_DIR"]
    )

    # -------------------- Accelerator --------------------
    # Default: no mixed precision (uses float32)
    # Default: don't split large batches to multi GPUs
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision, split_batches=args.split_batch
    )        

    # -------------------- Initialization --------------------
    t_start = datetime.now(pytz.timezone("Europe/Zurich"))
    # Resume previous run
    if resume_run is not None:
        print(f"Resume run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        # Resume config file
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
    else:
        # Load config
        if args.task_name == "MEF":
            config_path = "config/train/mef-train.yaml"
        elif args.task_name == "MFF":
            config_path = "config/train/mff-train.yaml"
        elif args.task_name == "IVF":
            config_path = "config/train/ivf-train.yaml"
        elif args.task_name == "MVF":
            config_path = "config/train/mvf-train.yaml"
        else:
            raise NotImplementedError(f"Unknown task: {args.task_name}")
        cfg = recursive_load_config(config_path, unknown_args)

        # Output folder name
        if args.add_datetime_prefix:
            job_name = (
                f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}"
                f"-{args.task_name}"
                f"-crop{cfg.augmentation.random_crop_hw[0]}"
                f"-bs{cfg.dataloader.effective_batch_size}_{cfg.dataloader.max_train_batch_size}"  # batch info
                f"-coef{'_'.join(map(str, cfg.loss.kwargs.coef))}"  # e.g., [1,2,3] -> "1_2_3"
                f"-lr{cfg.lr:.0e}"  # e.g., 0.0001 -> lr1e-4
            )
        else:
            job_name = (
                f"-{args.task_name}"
                f"-crop{cfg.augmentation.random_crop_hw[0]}"
                f"-bs{cfg.dataloader.effective_batch_size}_{cfg.dataloader.max_train_batch_size}"  # batch info
                f"-coef{'_'.join(map(str, cfg.loss.kwargs.coef))}"  # e.g., [1,2,3] -> "1_2_3"
                f"-lr{cfg.lr:.0e}"  # e.g., 0.0001 -> lr1e-4
            )

        if "no" != args.mixed_precision:
            job_name += f"_{args.mixed_precision}"

        # Output directory
        if output_dir is not None:
            out_dir_run = os.path.join(output_dir, job_name)
        else:
            out_dir_run = os.path.join("./output", job_name)
        if accelerator.is_main_process:
            os.makedirs(out_dir_run, exist_ok=False)

    # Other directories
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    out_dir_vis = os.path.join(out_dir_run, "visualization")
    if accelerator.is_main_process:
        if not os.path.exists(out_dir_ckpt):
            os.makedirs(out_dir_ckpt)
        if not os.path.exists(out_dir_tb):
            os.makedirs(out_dir_tb)
        if not os.path.exists(out_dir_eval):
            os.makedirs(out_dir_eval)
        if not os.path.exists(out_dir_vis):
            os.makedirs(out_dir_vis)
    accelerator.wait_for_everyone()

    # -------------------- Logging settings --------------------
    config_logging(cfg.logging, out_dir=out_dir_run)
    if accelerator.is_main_process:
        logging.info(f"start at {t_start}")
        logging.debug(f"args: {args}")
        logging.debug(f"config: {cfg}")
        logging.debug(
            f"accelerator: {accelerator.mixed_precision = }, {accelerator.split_batches = }"
        )
        assert accelerator.split_batches == args.split_batch

    # Initialize wandb
    if accelerator.is_main_process:
        if not args.no_wandb:
            if resume_run is not None:
                wandb_id = load_wandb_job_id(out_dir_run)
                wandb_cfg_dic = {
                    "id": wandb_id,
                    "resume": "must",
                    **cfg.wandb,
                }
            else:
                wandb_cfg_dic = {
                    "config": dict(cfg),
                    "name": job_name,
                    "mode": "online",
                    **cfg.wandb,
                }
            wandb_cfg_dic.update({"dir": out_dir_run})
            wandb_run = init_wandb(enable=True, **wandb_cfg_dic)
            save_wandb_job_id(wandb_run, out_dir_run)
        else:
            init_wandb(enable=False)

        # Tensorboard (should be initialized after wandb)
        tb_logger.set_dir(out_dir_tb)

        log_slurm_job_id(step=0)
    accelerator.wait_for_everyone()

    # -------------------- Device --------------------
    device = accelerator.device
    device_id = device.index if device.type == "cuda" else None
    device_id = 0 if device_id is None else device_id
    logging.info(f"{device = }, {device_id = }")
    n_gpu = accelerator.state.num_processes
    if accelerator.is_main_process:
        mixed_precision = accelerator.mixed_precision
        logging.info(f"{mixed_precision = }")
        logging.info(f"{n_gpu = }")

    # -------------------- Snapshot of code and config --------------------
    if resume_run is None:
        if accelerator.is_main_process:
            _output_path = os.path.join(out_dir_run, "config.yaml")
            with open(_output_path, "w+") as f:
                OmegaConf.save(config=cfg, f=f)
            logging.info(f"Config saved to {_output_path}")
            # Copy and archive code on the first run
            code_snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar.gz")
            logging.info("Saving code snapshot...")
            create_code_snapshot(code_snapshot_path, source_dir=".")
            logging.info(f"Code snapshot saved to: {code_snapshot_path}")
    accelerator.wait_for_everyone()

    # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size / n_gpu
    if args.split_batch:
        assert cfg.dataloader.max_train_batch_size >= n_gpu, (
            "not enough batch size to split"
        )
        assert 0 == cfg.dataloader.max_train_batch_size % n_gpu, (
            f"can't split {cfg.dataloader.max_train_batch_size} into {n_gpu} gpus"
        )
        accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size
    else:
        accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size / n_gpu
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)

    if accelerator.is_main_process:
        logging.info(
            f"Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}, on {n_gpu} devices"
        )

    # -------------------- Data --------------------
    init_loader_seed = cfg.dataloader.seed

    if init_loader_seed is None:
        loader_generator = None
    else:
        init_loader_seed += 1234321 * device_id  # account for multi-GPU randomness
        loader_generator = torch.Generator().manual_seed(init_loader_seed)

    # Training dataset
    cfg_train_data = OmegaConf.load(cfg.dataset_cfg.train)
    train_dataset: BaseTwoModalDataset = get_multi_frame_dataset(
        cfg_train_data,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        init_seed=init_loader_seed,
    )
    logging.debug("Augmentation: ", cfg.augmentation)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.dataloader.max_train_batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=True,
        generator=loader_generator,
        drop_last=True,
    )

    # -------------------- Model --------------------
    model = VideoFusion(model_config=cfg)

    # -------------------- Trainer --------------------
    # For Multi-Exposure Video Fusion
    if args.task_name == "MEF":
        from src.trainer.mef_trainer import MEFTrainer
        trainer = MEFTrainer(
            cfg=cfg,
            model=model,
            train_dataloader=train_loader,
            accelerator=accelerator,
            out_dir_ckpt=out_dir_ckpt,
            out_dir_eval=out_dir_eval,
            out_dir_vis=out_dir_vis,
            accumulation_steps=accumulation_steps,
            n_gpu=n_gpu,
        )
    # For Multi-Focus Video Fusion
    elif args.task_name =="MFF":
        from src.trainer.mff_trainer import MFFTrainer
        trainer = MFFTrainer(
            cfg=cfg,
            model=model,
            train_dataloader=train_loader,
            accelerator=accelerator,
            out_dir_ckpt=out_dir_ckpt,
            out_dir_eval=out_dir_eval,
            out_dir_vis=out_dir_vis,
            accumulation_steps=accumulation_steps,
            n_gpu=n_gpu,
        )
    # For Infrared-Visible Video Fusion
    elif args.task_name =="IVF":
        from src.trainer.ivf_trainer import IVFTrainer
        trainer = IVFTrainer(
            cfg=cfg,
            model=model,
            train_dataloader=train_loader,
            accelerator=accelerator,
            out_dir_ckpt=out_dir_ckpt,
            out_dir_eval=out_dir_eval,
            out_dir_vis=out_dir_vis,
            accumulation_steps=accumulation_steps,
            n_gpu=n_gpu,        
        )
    # For Medical Video Fusion
    elif args.task_name =="MVF":
        from src.trainer.mvf_trainer import MVFTrainer
        trainer = MVFTrainer(
            cfg=cfg,
            model=model,
            train_dataloader=train_loader,
            accelerator=accelerator,
            out_dir_ckpt=out_dir_ckpt,
            out_dir_eval=out_dir_eval,
            out_dir_vis=out_dir_vis,
            accumulation_steps=accumulation_steps,
            n_gpu=n_gpu,
        )
    else:
        raise NotImplementedError(f"Unknown task: {args.task_name}")

    # -------------------- Checkpoint --------------------
    if resume_run is not None and accelerator.is_main_process:
        trainer.load_checkpoint(
            resume_run, load_trainer_state=True, resume_lr_scheduler=True
        )

    # -------------------- Training & Evaluation Loop --------------------
    accelerator.wait_for_everyone()
    try:
        with accelerator.autocast():
            trainer.train()
    except Exception as e:
        logging.exception(e)
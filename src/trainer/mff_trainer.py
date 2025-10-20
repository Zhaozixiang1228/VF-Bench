# Last modified: 2025-10-20
import torch.nn.functional as F
import logging
import os
import shutil
from datetime import datetime
from typing import List, Union
import random
import pdb
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import einops
from accelerate import Accelerator
from torch.optim.adam import Adam
import time
from datetime import timedelta
import inspect


from src.model.net import VideoFusion
from src.util.data_loader import skip_first_batches
from src.util.logging_util import eval_dic_to_text, tb_logger
from src.util.seeding import generate_seed_sequence
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker, compute_metrics
from src.util.loss import get_loss
from src.util import metric
from src.util.io import pred_2_8bit, save_image


class MFFTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: VideoFusion,
        train_dataloader: DataLoader,
        accelerator: Accelerator,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        n_gpu: int,
        val_dataloaders: List[DataLoader] = None,
        random_far_near: bool = True,
        maskfusion: bool = True,
        downsample_train: bool = False,
    ):
        self.cfg: OmegaConf = cfg
        self.model: VideoFusion = model

        self.accelerator = accelerator
        self.device = accelerator.device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.accumulation_steps: int = accumulation_steps
        self.n_gpu: int = n_gpu
        self.random_far_near = random_far_near
        self.maskfusion= maskfusion
        self.downsample_train= downsample_train
        if self.downsample_train:
            logging.info("downsample_train to 128x128")
        # Optimizer: Adam
        lr = self.cfg.lr
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Eval metrics
        # (refer to https://github.com/markkua/VideoDepth/blob/bcd6d395ee93d008ef31a636ca082d70ee31d53d/config/multi_frame_exp/05_mv_overlap/51_tartanair_1-3frame_fixed.yaml#L85)
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(
            *[
                "loss",
                "loss_int",
                "loss_grad",
                "loss_ssim",
                "loss_temp",
            ]
        )

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    def _prepare_accelerator(self):
        if self.accelerator.mixed_precision == "no":
            weight_dtype = torch.float
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        elif self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        else:
            raise NotImplementedError(
                f"Not implemented for mixed_precission={self.accelerator.mixed_precision}"
            )

        self.model = self.model.to(self.accelerator.device, dtype=weight_dtype)

        logging.debug(
            f"{self.accelerator.device}: GPU memory allocated before accelerator.prepare(): {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )
        logging.debug(
            f"{self.accelerator.device}: GPU memory reserved before accelerator.prepare(): {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )

        # Call prepare()
        (
            self.model,
            self.optimizer,
            self.train_loader,
        ) = self.accelerator.prepare(self.model, self.optimizer, self.train_loader)
        logging.debug(
            f"{self.accelerator.device}: GPU memory allocated after accelerator.prepare(): {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )
        logging.debug(
            f"{self.accelerator.device}: GPU memory reserved after accelerator.prepare(): {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )

    def train(self, t_end=None):
        if self.accelerator.is_main_process:
            logging.info("Start training")

        device = self.device
        self._prepare_accelerator()
        self.train_metrics.reset()
        accumulated_step = 0
        iter_start_time = time.time()

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            if self.accelerator.is_main_process:
                logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.train()

                # Training step
                if self.random_far_near:
                    # Randomly change the order of the images
                    if random.random() < 0.5:
                        near_focus_img = batch["far"].to(self.device)
                        far_focus_img = batch["near"].to(self.device)
                    else:
                        far_focus_img = batch["far"].to(self.device)
                        near_focus_img = batch["near"].to(self.device)
                else:
                    # Use the original order of the images
                    far_focus_img = batch["far"].to(self.device)
                    near_focus_img = batch["near"].to(self.device)

                # Predict the noise residual
                if self.downsample_train:
                    far_focus_img=F.interpolate(far_focus_img, size=(3, 128, 128), mode='trilinear',align_corners=False)
                    near_focus_img=F.interpolate(near_focus_img, size=(3, 128, 128), mode='trilinear',align_corners=False)

                fusion_pred, flow_net = self.model(far_focus_img, near_focus_img)
                if torch.isnan(fusion_pred).any():
                    logging.warning(
                        f"device: {self.accelerator.device} model_pred contains NaN."
                    )

                # Loss
                loss_output = self.loss(
                    fusion_pred,  # B, 3, C, H, W
                    far_focus_img,  # B, 5, C, H, W
                    near_focus_img,  # B, 5, C, H, W
                    flow_net,
                )

                self.train_metrics.update("loss", loss_output["loss"].item())
                self.train_metrics.update("loss_int", loss_output["loss_int"].item())
                self.train_metrics.update("loss_grad", loss_output["loss_grad"].item())
                self.train_metrics.update("loss_ssim", loss_output["loss_ssim"].item())
                self.train_metrics.update("loss_temp", loss_output["loss_temp"].item())
                loss = loss_output["loss"] / self.gradient_accumulation_steps

                if self.cfg.trainer.get("loss_devided_by_n_gpu", False):
                    loss = loss / self.n_gpu
                self.accelerator.backward(loss)
                accumulated_step += 1

                self.n_batch_in_epoch += 1

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Calculate ETA
                    elapsed = time.time() - iter_start_time
                    avg_time_per_iter = elapsed / self.effective_iter
                    if self.max_iter > 0:
                        remaining_iter = self.max_iter - self.effective_iter
                        eta_seconds = int(avg_time_per_iter * remaining_iter)
                        eta_str = str(timedelta(seconds=eta_seconds))
                    else:
                        eta_str = "N/A"

                    train_loss_results = self.train_metrics.result()
                    for k, v in train_loss_results.items():
                        v_tensor = torch.tensor(v, device=device)
                        gathered = self.accelerator.gather(v_tensor)
                        gathered = torch.mean(gathered).item()

                        if self.accelerator.is_main_process:
                            tb_logger.writer.add_scalar(
                                f"train/{k}", gathered, global_step=self.effective_iter
                            )
                            train_loss_results[k] = gathered

                    if self.accelerator.is_main_process:
                        lr = self.lr_scheduler.get_last_lr()[0]
                        tb_logger.writer.add_scalar(
                            "lr", lr, global_step=self.effective_iter
                        )
                        tb_logger.writer.add_scalar(
                            "n_batch_in_epoch",
                            self.n_batch_in_epoch,
                            global_step=self.effective_iter,
                        )

                        # loss logging
                        loss_str = ", ".join(
                            f"{k}={v:.5f}" for k, v in train_loss_results.items()
                        )
                        logging.info(
                            f"iter {self.effective_iter:5d} (epoch {epoch:2d}): {loss_str}, lr={lr:.3e}, ETA={eta_str}"
                        )

                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.save_checkpoint(
                                ckpt_name=self._get_backup_ckpt_name(),
                                save_train_state=False,
                            )
                            logging.info("Training ended.")

                        self.accelerator.wait_for_everyone()
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.save_checkpoint(
                                ckpt_name="latest", save_train_state=True
                            )
                            logging.info("Time is up, training paused.")

                        self.accelerator.wait_for_everyone()
                        return

                    torch.cuda.empty_cache()

            # Epoch end
            self.n_batch_in_epoch = 0

    def _train_step_callback(self):
        """Executed after every iteration"""
        self.accelerator.wait_for_everyone()

        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            if self.accelerator.is_main_process:
                self.save_checkpoint(
                    ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                )

        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and self.accelerator.is_main_process
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            if self.accelerator.is_main_process:
                logging.info(
                    f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
                )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        os.makedirs(ckpt_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(ckpt_dir, "model.pth")
        model_unwrap = self.accelerator.unwrap_model(self.model)
        torch.save(model_unwrap.state_dict(), model_path)
        logging.info(f"Model is saved to: {model_path}")

        if save_train_state:
            optimizer = self.accelerator.unwrap_model(self.optimizer)
            state = {
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            with open(os.path.join(ckpt_dir, "iter.txt"), "w+") as f:
                f.write(self._get_backup_ckpt_name())

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")

        # Load model
        model_path = os.path.join(ckpt_path, "model.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logging.info(f"Model parameters are loaded from {model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(
                os.path.join(ckpt_path, "trainer.ckpt"), weights_only=False
            )
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"

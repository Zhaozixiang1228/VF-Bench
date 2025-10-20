# Last modified: 2025-10-18

import logging
import os
import sys
import wandb
from tabulate import tabulate

import tarfile
import subprocess
import tempfile

from torch.utils.tensorboard import SummaryWriter


def config_logging(cfg_logging, out_dir=None):
    file_level = cfg_logging.get("file_level", 10)
    console_level = cfg_logging.get("console_level", 10)

    log_formatter = logging.Formatter(cfg_logging["format"])

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_logger.setLevel(min(file_level, console_level))

    if out_dir is not None:
        _logging_file = os.path.join(
            out_dir, cfg_logging.get("filename", "logging.log")
        )
        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Avoid pollution by packages
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)


class MyTrainingLogger:
    """Tensorboard + wandb logger"""

    writer: SummaryWriter
    is_initialized = False

    def __init__(self) -> None:
        pass

    def set_dir(self, tb_log_dir):
        if self.is_initialized:
            raise ValueError("Do not initialize writer twice")
        self.writer = SummaryWriter(tb_log_dir)
        self.is_initialized = True

    def log_dic(self, scalar_dic, global_step, walltime=None):
        for k, v in scalar_dic.items():
            self.writer.add_scalar(k, v, global_step=global_step, walltime=walltime)
        return


# global instance
tb_logger = MyTrainingLogger()


def create_code_snapshot(output_file: str, source_dir: str = ".") -> bool:
    """
    Create a tar.gz code snapshot using rsync, respecting .gitignore rules.

    Args:
        source_dir: Source directory to snapshot (defaults to current directory)
        output_file: Path for the output tar.gz file (if None, generates a timestamped name)

    Returns:
        bool: True if snapshot was created successfully, False otherwise
    """
    try:
        source_dir = os.path.abspath(source_dir)

        if not os.path.exists(source_dir):
            raise ValueError(f"Source directory does not exist: {source_dir}")

        output_file = os.path.abspath(output_file)

        # Create temporary directory for rsync
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create rsync snapshot in temp directory
            rsync_command = [
                "rsync",
                "-a",  # Archive mode
                "-v",  # Verbose
                "--delete",  # Delete extraneous files
                "--exclude=.git/",  # Exclude .git directory
                "--filter=:- .gitignore",  # Use .gitignore as filter
                f"{source_dir}/",  # Source with trailing slash
                temp_dir,  # Temporary destination
            ]

            # Execute rsync
            result = subprocess.run(rsync_command, capture_output=True, text=True)

            if result.returncode != 0:
                logging.error(f"Rsync failed: {result.stderr}")
                return False

            # Create tar.gz archive
            temp_archive = output_file + ".tmp"
            try:
                with tarfile.open(temp_archive, "w:gz", compresslevel=6) as tar:
                    # Change to temp directory so paths in archive are relative
                    original_dir = os.getcwd()
                    os.chdir(temp_dir)
                    try:
                        # Add all contents to archive
                        for root, dirs, files in os.walk("."):
                            for file in files:
                                file_path = os.path.join(root, file)
                                tar.add(file_path)
                    finally:
                        os.chdir(original_dir)

                # Atomic replacement of final file
                if os.path.exists(output_file):
                    os.remove(output_file)
                os.rename(temp_archive, output_file)

                logging.info(f"Snapshot archive created successfully at {output_file}")
                return True

            except Exception as e:
                if os.path.exists(temp_archive):
                    os.remove(temp_archive)
                raise e

    except subprocess.SubprocessError as e:
        logging.error(f"Failed to execute rsync: {e}")
        return False
    except Exception as e:
        logging.error(f"Failed to create snapshot: {e}")
        return False


# -------------- wandb tools --------------
def init_wandb(enable: bool, **kwargs):
    if enable:
        run = wandb.init(sync_tensorboard=True, **kwargs)
    else:
        run = wandb.init(mode="disabled")
    return run


def log_slurm_job_id(step):
    global tb_logger
    _jobid = os.getenv("SLURM_JOB_ID")
    if _jobid is None:
        _jobid = -1
    tb_logger.writer.add_scalar("job_id", int(_jobid), global_step=step)
    logging.debug(f"Slurm job_id: {_jobid}")


def load_wandb_job_id(out_dir):
    with open(os.path.join(out_dir, "WANDB_ID"), "r") as f:
        wandb_id = f.read()
    return wandb_id


def save_wandb_job_id(run, out_dir):
    with open(os.path.join(out_dir, "WANDB_ID"), "w+") as f:
        f.write(run.id)


def eval_dic_to_text(val_metrics: dict, msg: str = "") -> str:
    eval_text = f"Evaluation metrics:\n{msg}\n"

    eval_text += tabulate([val_metrics.keys(), val_metrics.values()])
    return eval_text

def setup_logging(log_dir, file_log_level="DEBUG", console_log_level="INFO"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "test.log")

    # Clear existing handlers to prevent duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure basic logging with two handlers
    logging.basicConfig(
        level=getattr(logging, file_log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info(f"Logging system initialized. Log file: {log_file}")
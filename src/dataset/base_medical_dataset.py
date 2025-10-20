# Author: Zixiang Zhao
# Last modified: 2025-04-11

import os
import sys

sys.path.append(os.getcwd())
from src.dataset.base_two_modal_dataset import BaseTwoModalDataset, DatasetMode

rgb_ME_filename_col_names_dict = {
    DatasetMode.TRAIN: ["mri", "other"],
    DatasetMode.EVAL: ["mri", "other"],
    DatasetMode.TEST: ["mri", "other"],
}


class BaseMVDataset(BaseTwoModalDataset):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            filename_col_names=rgb_ME_filename_col_names_dict,
            **kwargs,
        )


if "__main__" == __name__:
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.dataset import get_multi_frame_dataset

    import torchshow as ts

    data_cfg = OmegaConf.load("config/dataset/MVF/Harvard/harvard_5-frame.yaml")

    dataset = get_multi_frame_dataset(
        data_cfg, base_data_dir="data", mode=DatasetMode.TRAIN
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    for batch in tqdm(dataloader):
        # under = batch["under"]
        print(batch.keys())
        print(batch["data_path_ls_dict"])
        print(batch["mri"].shape)
        print(batch["other"].shape)
        ts.save(batch["mri"][0, :, :, :, :], "output/debug/mri.jpg")
        ts.save(batch["other"][0, :, :, :, :], "output/debug/other.jpg")

        break

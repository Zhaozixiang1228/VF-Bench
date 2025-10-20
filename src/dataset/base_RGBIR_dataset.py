# Author: Zixiang Zhao
# Last modified: 2025-10-20

import os
import sys

sys.path.append(os.getcwd())
from src.dataset.base_two_modal_dataset import BaseTwoModalDataset, DatasetMode

rgb_ir_filename_col_names_dict = {
    DatasetMode.TRAIN: ["ir", "rgb"],
    DatasetMode.EVAL: ["ir", "rgb"],
    DatasetMode.TEST: ["ir", "rgb"],
}


class BaseRGBIRDataset(BaseTwoModalDataset):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            filename_col_names=rgb_ir_filename_col_names_dict,
            **kwargs,
        )

if "__main__" == __name__:
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.dataset import get_multi_frame_dataset

    import torchshow as ts

    data_cfg = OmegaConf.load("config/dataset/IVF/VTMOT/vtmot_5-frame.yaml")

    dataset = get_multi_frame_dataset(
        data_cfg, base_data_dir="data", mode=DatasetMode.TRAIN
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    for batch in tqdm(dataloader):
        print(batch.keys())
        print(batch["data_path_ls_dict"])
        print(batch["ir"].shape)
        print(batch["rgb"].shape)
        ts.save(batch["ir"][0, :, :, :, :], "output/debug/ir.jpg")
        ts.save(batch["rgb"][0, :, :, :, :], "output/debug/rgb.jpg")

        # break

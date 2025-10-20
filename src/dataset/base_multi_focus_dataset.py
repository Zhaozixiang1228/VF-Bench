# Author: Zixiang Zhao
# Last modified: 2025-10-20

import os
import sys

sys.path.append(os.getcwd())
from src.dataset.base_two_modal_dataset import BaseTwoModalDataset, DatasetMode

rgb_ME_filename_col_names_dict = {
    DatasetMode.TRAIN: ["far", "near"],
    DatasetMode.EVAL: ["far", "near"],
    DatasetMode.TEST: ["far", "near"],
}


class BaseMFDataset(BaseTwoModalDataset):
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

    data_cfg = OmegaConf.load("config/dataset/MFF/DAVIS/davis_5-frame.yaml")

    dataset = get_multi_frame_dataset(
        data_cfg, base_data_dir="data", mode=DatasetMode.TRAIN
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    i = 0
    for batch in tqdm(dataloader):
        i += 1
        if i > 20:
            break
        print(batch.keys())
        print(batch["data_path_ls_dict"])
        print(batch["far"].shape)
        print(batch["near"].shape)
        ts.save(batch["far"][0, :, :, :, :], "output/debug/far" + str(i) + ".jpg")
        ts.save(batch["near"][0, :, :, :, :], "output/debug/near" + str(i) + ".jpg")

        # break

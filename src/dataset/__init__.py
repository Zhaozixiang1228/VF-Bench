# Author: Zixiang Zhao
# Last modified: 2025-10-05
# --------------------------------------------------------------------------


import os
from omegaconf import OmegaConf

from .base_two_modal_dataset import BaseTwoModalDataset, DatasetMode
from .base_RGBIR_dataset import BaseRGBIRDataset
from .base_multi_expo_dataset import BaseMEDataset
from .base_medical_dataset import BaseMVDataset
from .base_multi_focus_dataset import BaseMFDataset

dataset_name_class_dict = {
    "vtmot_dataset": BaseRGBIRDataset,
    "youtube_dataset": BaseMEDataset,
    "harvard_dataset": BaseMVDataset,
    "davis_dataset": BaseMFDataset,
}


def get_multi_frame_dataset(
    cfg_data_split: OmegaConf, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseTwoModalDataset:
    if "mixed" == cfg_data_split.class_name:
        assert DatasetMode.TRAIN == mode, "Only training mode supports mixed datasets."
        dataset_ls = [
            get_multi_frame_dataset(_cfg, base_data_dir, mode, **kwargs)
            for _cfg in cfg_data_split.dataset_list
        ]
        return dataset_ls
    elif cfg_data_split.class_name in dataset_name_class_dict.keys():
        # Proper deep copy to avoid modifying original config
        cfg_dataset = OmegaConf.create(
            OmegaConf.to_container(cfg_data_split, resolve=True)
        )
        dataset_class = dataset_name_class_dict[cfg_dataset.pop("class_name")]
        dataset = dataset_class(
            mode=mode,
            dataset_dir=os.path.join(base_data_dir, cfg_dataset.pop("dir")),
            **cfg_dataset,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset

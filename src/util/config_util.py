# Last modified: 2025-09-27

import omegaconf
from omegaconf import OmegaConf
from typing import List

def recursive_load_config(config_path: str, cli_args: List[str] = None) -> OmegaConf:
    # Overall override order: base_config < current config < CLI
    # 1. Load the current configuration
    conf = OmegaConf.load(config_path)
    output_conf = OmegaConf.create({})
    # 2. If there is a base_config, load it recursively
    base_configs = conf.get("base_config", default_value=None)
    if base_configs is not None:
        assert isinstance(base_configs, omegaconf.ListConfig)
        for _path in base_configs:
            assert _path != config_path, (
                "Circular merging detected: base_config should not include itself."
            )
            _base_conf = recursive_load_config(_path)
            output_conf = OmegaConf.merge(output_conf, _base_conf)
    # 3. Merge the current level configuration
    output_conf = OmegaConf.merge(output_conf, conf)
    # 4. Apply CLI overrides (with validation)
    if cli_args:
        cli_conf = OmegaConf.from_dotlist(cli_args)
        # Only allow overriding existing keys
        for k in cli_conf.keys():
            if not OmegaConf.select(output_conf, k, default=None):
                raise KeyError(f"Argument '{k}' not defined in YAML config (from CLI).")
        output_conf = OmegaConf.merge(output_conf, cli_conf)
    return output_conf


if "__main__" == __name__:
    conf = recursive_load_config("config/train/mvf-train.yaml")
    print(OmegaConf.to_yaml(conf))
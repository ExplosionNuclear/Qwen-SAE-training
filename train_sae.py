from dotenv import load_dotenv
from typing import Any
from omegaconf import OmegaConf
import fire

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner
from sae_training.utils import _parse_dtype, _parse_device, _parse_hook_point_layer, get_hub_repo_id, get_project_name


def build_config(config: str = "configs/train.yaml", **overrides: Any) -> LanguageModelSAERunnerConfig:
    base_cfg = OmegaConf.load(config)
    override_cfg = OmegaConf.create(overrides) if overrides else OmegaConf.create({})
    override_cfg.hook_point_layer = _parse_hook_point_layer(override_cfg.hook_point_layer)
    merged: Any = OmegaConf.merge(base_cfg, override_cfg)
    merged.hub_repo_id = get_hub_repo_id(merged.model_name, merged.hook_point)
    merged.wandb_project = get_project_name(merged.model_name, merged.hook_point)

    print(merged.hook_point_layer)
    # Map dtype and device
    dtype_str = merged.get("dtype", "float32")
    device_str = merged.get("device", "auto")

    data: dict[str, Any] = OmegaConf.to_container(merged, resolve=True)  # type: ignore[assignment]
    data["dtype"] = _parse_dtype(dtype_str)
    data["device"] = _parse_device(device_str)

    return LanguageModelSAERunnerConfig(**data)


def train(config: str = "configs/train.yaml", **overrides: Any):
    cfg = build_config(config, **overrides)
    language_model_sae_runner(cfg)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(train)
    
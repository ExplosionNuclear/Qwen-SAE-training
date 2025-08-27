from typing import Any, cast

import os
import wandb
from clearml import Task

from sae_training.config import LanguageModelSAERunnerConfig

# from sae_training.activation_store import ActivationStore
from sae_training.train_sae_on_language_model import train_sae_on_language_model
from sae_training.utils import LMSparseAutoencoderSessionloader


def language_model_sae_runner(cfg: LanguageModelSAERunnerConfig):
    """ """

    # Initialize logging backend
    clearml_task = None
    if cfg.logger_backend == "wandb" and cfg.log_to_wandb:
        if cfg.wandb_api_key:
            wandb.login(key=cfg.wandb_api_key)
        wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name)
    elif cfg.logger_backend == "clearml":
        clearml_task = Task.init(
            project_name=cfg.clearml_project or cfg.wandb_project,
            task_name=cfg.clearml_task_name or (cfg.run_name or "sae-training"),
            tags=cfg.clearml_tags,
            output_uri=None,
        )

    if cfg.from_pretrained_path is not None:
        (
            model,
            sparse_autoencoder,
            activations_loader,
        ) = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
            cfg.from_pretrained_path
        )
        cfg = sparse_autoencoder.cfg
    else:
        loader = LMSparseAutoencoderSessionloader(cfg)
        model, sparse_autoencoder, activations_loader = loader.load_session()

    # train SAE
    sparse_autoencoder = train_sae_on_language_model(
        model,
        sparse_autoencoder,
        activations_loader,
        n_checkpoints=cfg.n_checkpoints,
        batch_size=cfg.train_batch_size,
        feature_sampling_window=cfg.feature_sampling_window,
        dead_feature_threshold=cfg.dead_feature_threshold,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency,
    )

    if cfg.logger_backend == "wandb" and cfg.log_to_wandb:
        wandb.finish()
    elif cfg.logger_backend == "clearml" and clearml_task is not None:
        clearml_task.close()

    return sparse_autoencoder

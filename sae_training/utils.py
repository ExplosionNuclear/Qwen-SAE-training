from typing import Any, Tuple

import torch
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore
from sae_training.sae_group import SAEGroup
from sae_training.sparse_autoencoder import SparseAutoencoder


class LMSparseAutoencoderSessionloader:
    """
    Responsible for loading all required
    artifacts and files for training
    a sparse autoencoder on a language model
    or analysing a pretraining autoencoder
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg

    def load_session(
        self,
    ) -> Tuple[HookedTransformer, SAEGroup, ActivationsStore]:
        """
        Loads a session for training a sparse autoencoder on a language model.
        """
        if isinstance(self.cfg.model, HookedTransformer):
            model = self.cfg.model
        else:
            model = self.get_model(self.cfg.model_name)

        model.to(self.cfg.device)
        activations_loader = self.get_activations_loader(self.cfg, model)
        sparse_autoencoder = self.initialize_sparse_autoencoder(self.cfg)

        return model, sparse_autoencoder, activations_loader

    @classmethod
    def load_session_from_pretrained(
        cls, path: str, cfg_overrides: dict[str, Any] | None = None
    ) -> Tuple[HookedTransformer, SAEGroup, ActivationsStore]:
        """
        Loads a session for analysing a pretrained sparse autoencoder group.
        """
        loaded = SAEGroup.load_from_pretrained(path)

        # Helper to build model and activations loader without re-initializing SAEGroup
        def _init_model_and_acts(cfg: Any) -> tuple[HookedTransformer, ActivationsStore]:
            loader = cls(cfg)
            if isinstance(cfg.model, HookedTransformer):
                model_local = cfg.model
            else:
                model_local = loader.get_model(cfg.model_name)
            model_local.to(cfg.device)
            activations_loader_local = loader.get_activations_loader(cfg, model_local)
            return model_local, activations_loader_local

        if isinstance(loaded, dict):
            cfg = loaded["cfg"]
            if cfg_overrides:
                for k, v in cfg_overrides.items():
                    setattr(cfg, k, v)
            # Build a single SAE and wrap it into a lightweight SAEGroup without constructing new ones
            ae = SparseAutoencoder(cfg=cfg)
            ae.load_state_dict(loaded["state_dict"])
            group = SAEGroup.__new__(SAEGroup)
            group.cfg = cfg
            group.autoencoders = [ae]
            model, activations_loader = _init_model_and_acts(cfg)
            return model, group, activations_loader
        elif isinstance(loaded, SAEGroup):
            cfg = loaded.cfg
            if cfg_overrides:
                for k, v in cfg_overrides.items():
                    setattr(cfg, k, v)
            model, activations_loader = _init_model_and_acts(cfg)
            return model, loaded, activations_loader
        else:
            raise ValueError(
                "The loaded sparse_autoencoders object is neither an SAE dict nor a SAEGroup"
            )

    def get_model(self, model_name: str):
        """
        Loads a model from transformer lens
        """

        # Todo: add check that model_name is valid

        model = HookedTransformer.from_pretrained(model_name)

        return model

    def initialize_sparse_autoencoder(self, cfg: Any):
        """
        Initializes a sparse autoencoder group, which contains multiple sparse autoencoders
        """

        sparse_autoencoder = SAEGroup(cfg)

        return sparse_autoencoder

    def get_activations_loader(self, cfg: Any, model: HookedTransformer):
        """
        Loads a DataLoaderBuffer for the activations of a language model.
        """

        activations_loader = ActivationsStore(
            cfg,
            model,
        )

        return activations_loader


def shuffle_activations_pairwise(datapath: str, buffer_idx_range: Tuple[int, int]):
    """
    Shuffles two buffers on disk.
    """
    assert (
        buffer_idx_range[0] < buffer_idx_range[1] - 1
    ), "buffer_idx_range[0] must be smaller than buffer_idx_range[1] by at least 1"

    buffer_idx1 = torch.randint(buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    buffer_idx2 = torch.randint(buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    while buffer_idx1 == buffer_idx2:  # Make sure they're not the same
        buffer_idx2 = torch.randint(
            buffer_idx_range[0], buffer_idx_range[1], (1,)
        ).item()

    buffer1 = torch.load(f"{datapath}/{buffer_idx1}.pt")
    buffer2 = torch.load(f"{datapath}/{buffer_idx2}.pt")
    joint_buffer = torch.cat([buffer1, buffer2])

    # Shuffle them
    joint_buffer = joint_buffer[torch.randperm(joint_buffer.shape[0])]
    shuffled_buffer1 = joint_buffer[: buffer1.shape[0]]
    shuffled_buffer2 = joint_buffer[buffer1.shape[0] :]

    # Save them back
    torch.save(shuffled_buffer1, f"{datapath}/{buffer_idx1}.pt")
    torch.save(shuffled_buffer2, f"{datapath}/{buffer_idx2}.pt")

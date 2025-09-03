import dataclasses
import gzip
import os
import pickle
from itertools import product
from typing import Any, Iterator
from types import SimpleNamespace

import torch

from sae_training.sparse_autoencoder import SparseAutoencoder
from safetensors.torch import load_file as safetensors_load_file


class SAEGroup:

    autoencoders: list[SparseAutoencoder]

    def __init__(self, cfg: Any):
        self.cfg = cfg
        # This will store tuples of (instance, hyperparameters)
        self.autoencoders = []
        self._init_autoencoders(cfg)

    def _init_autoencoders(self, cfg: Any):
        # Dynamically get all combinations of hyperparameters from cfg
        # Extract all hyperparameter lists from cfg
        hyperparameters = {k: v for k, v in vars(
            cfg).items() if isinstance(v, list)}
        if len(hyperparameters) > 0:
            keys, values = zip(*hyperparameters.items())
        else:
            # Ensure product(*values) yields one combination
            keys, values = (), ([()],)
        # Create all combinations of hyperparameters
        for combination in product(*values):
            params = dict(zip(keys, combination))
            cfg_copy = dataclasses.replace(cfg, **params)
            cfg_copy.__post_init__()
            # Insert the layer into the hookpoint
            cfg_copy.hook_point = cfg_copy.hook_point.format(
                layer=cfg_copy.hook_point_layer
            )
            # Create and store both the SparseAutoencoder instance and its parameters
            self.autoencoders.append(SparseAutoencoder(cfg_copy))

    def __iter__(self) -> Iterator[SparseAutoencoder]:
        # Make SAEGroup iterable over its SparseAutoencoder instances and their parameters
        for ae in self.autoencoders:
            yield ae  # Yielding as a tuple

    def __len__(self):
        # Return the number of SparseAutoencoder instances
        return len(self.autoencoders)

    def to(self, device: torch.device | str):
        for ae in self.autoencoders:
            ae.to(device)

    @classmethod
    def load_from_pretrained(cls, path: str):
        """
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        """

        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        # Load the state dictionary
        if path.endswith(".pt"):
            try:
                if torch.backends.mps.is_available():
                    group = torch.load(path, map_location="mps")
                    group["cfg"].device = "mps"
                else:
                    group = torch.load(path)
            except Exception as e:
                raise IOError(
                    f"Error loading the state dictionary from .pt file: {e}")

        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, "rb") as f:
                    group = pickle.load(f)
            except Exception as e:
                raise IOError(
                    f"Error loading the state dictionary from .pkl.gz file: {e}"
                )
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    group = pickle.load(f)
            except Exception as e:
                raise IOError(
                    f"Error loading the state dictionary from .pkl file: {e}")
        elif path.endswith(".safetensors"):
            tensors = safetensors_load_file(path, device="cpu")

            indices: set[int] = set()
            for key in tensors.keys():
                if "." not in key:
                    continue
                idx_str, _ = key.split(".", 1)
                try:
                    indices.add(int(idx_str))
                except ValueError:
                    continue

            # Helper to build a lightweight config without triggering heavy __post_init__
            def _make_lite_cfg(d_in: int, d_sae: int, dtype: torch.dtype, device: str | torch.device):
                cfg = SimpleNamespace()
                cfg.d_in = d_in
                cfg.d_sae = d_sae
                cfg.dtype = dtype
                cfg.device = device
                # Defaults matching LanguageModelSAERunnerConfig
                cfg.l1_coefficient = 1e-3
                cfg.lp_norm = 1
                cfg.use_ghost_grads = False
                cfg.hook_point = ""
                cfg.hook_point_head_index = None
                cfg.model_name = ""
                cfg.log_to_wandb = False
                cfg.expansion_factor = d_sae // d_in if d_in else 1
                return cfg

            # Infer minimal cfgs from tensor shapes
            inferred_ae_cfgs: list[Any] = []
            for idx in sorted(indices):
                w_dec = tensors.get(f"{idx}.W_dec")
                w_enc = tensors.get(f"{idx}.W_enc")
                if w_dec is None or w_enc is None:
                    raise ValueError(
                        f"Missing expected tensors for autoencoder index {idx}")
                d_sae = int(w_dec.shape[0])
                d_in = int(w_dec.shape[1])
                inferred_ae_cfgs.append(_make_lite_cfg(
                    d_in, d_sae, w_dec.dtype, "cpu"))

            # Create group instance without running __init__ (avoids expensive autoencoder construction)
            instance = cls.__new__(cls)
            # Store a minimal cfg; session code may supply a richer cfg via overrides
            instance.cfg = inferred_ae_cfgs[0]
            instance.autoencoders = []

            # Reconstruct each SAE and load its tensors using the lightweight cfgs
            for idx, cfg in enumerate(inferred_ae_cfgs):
                ae = SparseAutoencoder(cfg)
                state_dict_i: dict[str, torch.Tensor] = {}
                for k, v in tensors.items():
                    if k.startswith(f"{idx}."):
                        sub_key = k.split(".", 1)[1]
                        state_dict_i[sub_key] = v
                ae.load_state_dict(state_dict_i, strict=True)
                instance.autoencoders.append(ae)

            return instance
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, .pkl.gz and .safetensors"
            )

        return group

    def save_model(self, path: str) -> list[str]:
        """
        Basic save function for the model. Saves each autoencoder in a separate file.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        saved_files = []

        if path.endswith(".safetensors"):
            from safetensors.torch import save_file
            # Save each autoencoder in a separate file
            for ae in self.autoencoders:
                # Create layer-specific filename
                layer = ae.cfg.hook_point_layer
                # Get the last part (e.g., 'hook_resid_pre')
                hook_point = ae.cfg.hook_point.split('.')[-1]
                # Replace slashes for filesystem compatibility
                model_name = ae.cfg.model_name.replace('/', '-')

                # Generate individual file path
                individual_path = os.path.join(
                    folder, f"{model_name}_layer-{layer}.{hook_point}_{ae.cfg.d_sae}.safetensors")

                # Save individual autoencoder
                tensors = {k: v.detach().cpu()
                           for k, v in ae.state_dict().items()}
                save_file(tensors, individual_path)
                saved_files.append(individual_path)

        elif path.endswith(".pt"):
            # Save each autoencoder in a separate .pt file
            for ae in self.autoencoders:
                layer = ae.cfg.hook_point_layer
                hook_point = ae.cfg.hook_point.split('.')[-1]
                model_name = ae.cfg.model_name.replace('/', '-')

                individual_path = os.path.join(
                    folder, f"{model_name}_layer-{layer}.{hook_point}_{ae.cfg.d_sae}.pt")

                torch.save(
                    {"cfg": ae.cfg, "state_dict": ae.state_dict()}, individual_path)
                saved_files.append(individual_path)

        elif path.endswith("pkl.gz"):
            # Save each autoencoder in a separate .pkl.gz file
            for ae in self.autoencoders:
                layer = ae.cfg.hook_point_layer
                hook_point = ae.cfg.hook_point.split('.')[-1]
                model_name = ae.cfg.model_name.replace('/', '-')

                individual_path = os.path.join(
                    folder, f"{model_name}_layer-{layer}.{hook_point}_{ae.cfg.d_sae}.pkl.gz")

                with gzip.open(individual_path, "wb") as f:
                    pickle.dump(
                        {"cfg": ae.cfg, "state_dict": ae.state_dict()}, f)
                saved_files.append(individual_path)
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .safetensors, .pt and .pkl.gz"
            )

        # Print all saved files
        for saved_file in saved_files:
            print(f"Saved model to {saved_file}")

        return saved_files

    def get_name(self):
        layers = self.cfg.hook_point_layer
        if not isinstance(layers, list):
            layers = [layers]

        if len(layers) > 1:
            layer_string = f"{layers[0]}-{layers[-1]}"
        else:
            layer_string = f"{layers[0]}"
        sae_name = f"sae_group_{self.cfg.model_name}_{self.cfg.hook_point.format(layer=layer_string)}_{self.cfg.d_sae}"
        return sae_name

    def eval(self):
        for ae in self.autoencoders:
            ae.eval()

    def train(self):
        for ae in self.autoencoders:
            ae.train()

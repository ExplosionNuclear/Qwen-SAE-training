from typing import Any, Dict, Tuple

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, convert_llama_weights


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
        self
    ) -> Tuple[HookedTransformer, SAEGroup, ActivationsStore]:
        """
        Loads a session for training a sparse autoencoder on a language model.
        """
        model = self.get_model(self.cfg.model_name)

        model.to(self.cfg.device)
        self.cfg.d_in = self.cfg.d_in if self.cfg.d_in is not None else model.cfg.d_model
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
            model_local = loader.get_model(cfg.model_name)
            model_local.to(cfg.device)
            activations_loader_local = loader.get_activations_loader(
                cfg, model_local)
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

    def get_model(self, model_name: str, kwargs: Dict[str, Any] = {}):
        """
        Loads a model from transformer lens
        """

        # Todo: add check that model_name is valid

        print("model_name", model_name)

        if model_name not in OFFICIAL_MODEL_NAMES:
            return get_custom_hf_model(model_name, kwargs)

        return HookedTransformer.from_pretrained(model_name)

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

    buffer_idx1 = torch.randint(
        buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    buffer_idx2 = torch.randint(
        buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
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
    shuffled_buffer2 = joint_buffer[buffer1.shape[0]:]

    # Save them back
    torch.save(shuffled_buffer1, f"{datapath}/{buffer_idx1}.pt")
    torch.save(shuffled_buffer2, f"{datapath}/{buffer_idx2}.pt")


def get_custom_hf_model(model_name: str, kwargs: Dict[str, Any] = {}) -> HookedTransformer: 
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        **kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    hf_config = hf_model.config
    
    cfg_dict = {
        "d_model": hf_config.hidden_size,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_heads": hf_config.num_attention_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": min(hf_config.max_position_embeddings, 2048),
        "eps": getattr(hf_config, 'rms_norm_eps', 1e-6),
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.hidden_act,
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "rotary_adjacent_pairs": False,
        "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
        "final_rms": True,
        "gated_mlp": True,
        "model_name": model_name.split("/")[-1],
        "init_weights": False,
        "device": "cpu",
        "dtype": torch.float32,
    }
    
    if hasattr(hf_config, 'num_key_value_heads') and hf_config.num_key_value_heads != hf_config.num_attention_heads:
        cfg_dict["n_key_value_heads"] = hf_config.num_key_value_heads
    
    if hasattr(hf_config, 'rope_theta'):
        cfg_dict["rotary_base"] = hf_config.rope_theta
        print(f"Included rotary_base = {hf_config.rope_theta}")
    
    
    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    
    for param in hf_model.parameters():
        param.requires_grad = False
    
    state_dict = convert_llama_weights(hf_model, cfg)
    model = HookedTransformer(cfg)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print(f"Loading weights:")
    print(f"  - Missing keys: {len(missing_keys)}")
    print(f"  - Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        print(f"First 5 missing keys:")
        for i, key in enumerate(missing_keys[:5]):
            print(f"    {i+1}. {key}")
    
    if unexpected_keys:
        print(f"First 5 unexpected keys:")
        for i, key in enumerate(unexpected_keys[:5]):
            print(f"    {i+1}. {key}")
    
    model.set_tokenizer(tokenizer)
    
    print("✓ Model created and weights loaded!")
    
    return model

def _parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
    }
    return mapping.get(dtype_str.lower(), torch.float32)


def _parse_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _parse_hook_point_layer(hook_point_layer: Any) -> list[int]:
    hook_point_layer = list(hook_point_layer)
    if isinstance(hook_point_layer, list):
        return list(range(int(hook_point_layer[0]), int(hook_point_layer[-1]) + 1))
    else:
        return [hook_point_layer]


def get_hub_repo_id(model_name: str, hook_point: str) -> str:
    return f"Lucid-Layers-Inc/{model_name.split('/')[-1]}-{hook_point.split('.')[-1]}-SAE"


def get_project_name(model_name: str, hook_point: str) -> str:
    return f"mats_sae_training_{model_name.split('/')[-1]}_{hook_point.split('.')[-1]}-SAE"

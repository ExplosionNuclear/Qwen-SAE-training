import torch
from dotenv import load_dotenv
import sys

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

def main():
    load_dotenv()
    hook_point_layer = list(range(5))
    if len(sys.argv) >= 3: 
        start, end = int(sys.argv[1]), int(sys.argv[2])
        hook_point_layer = list(range(start, end + 1))
    
    print(f"Hook point layer: {hook_point_layer}")
    
    cfg = LanguageModelSAERunnerConfig(
        model_name="ExplosionNuclear/Llama-2.3-3B-Instruct-special",
        hook_point="blocks.{layer}.hook_resid_pre",
        hook_point_layer=hook_point_layer,
        dataset_path="ashaba1in/small_openwebtext",
        is_dataset_tokenized=False,
        context_size=128,
        d_in=3072,

        # SAE
        expansion_factor=6,  # d_sae = d_in * expansion_factor
        b_dec_init_method="mean",

        # Тренировка
        lr=3e-4,
        l1_coefficient=1e-3,
        lr_scheduler_name="constantwithwarmup",
        lr_warm_up_steps=1000,
        train_batch_size=4096*8,         
        n_batches_in_buffer=20,
        total_training_tokens=2_000_000,  
        store_batch_size=16,    

        wandb_project="mats_sae_training_llama32",
        wandb_log_frequency=10,
        wandb_api_key="a89e0ceef33f3c2cc4b7d9d9d5795fa238b4a60c",
        wandb_entity="rokser9-lucid-layers",
        eval_every_n_steps=80,

        logger_backend="clearml",
        
        n_checkpoints=4,
        checkpoint_path="checkpoints",

        push_to_hub=True,
        hub_repo_id="Lucid-Layers-Inc/llama23-sae-resid_pre",
        hub_private=False,

        # Прочее
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        dtype=torch.float32,
    )

    language_model_sae_runner(cfg)


if __name__ == "__main__":
    main()
    
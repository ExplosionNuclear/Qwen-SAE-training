#!/usr/bin/env python3

from safetensors.torch import load_file

path = "/root/.cache/huggingface/hub/models--Lucid-Layers-Inc--llama23-sae-resid_pre/snapshots/d4acf4f3408a8c8c3aa67f9223ee969c50c83878/Llama-2.3-3B-Instruct-special_blocks.5.hook_resid_pre_3072.safetensors"

try:
    tensors = load_file(path, device='cpu')
    print(f"Found {len(tensors)} tensors in file:")
    for k in sorted(tensors.keys()):
        print(f"  {k}: {tensors[k].shape}")
    
    # Look for patterns to understand the structure
    indices = set()
    for key in tensors.keys():
        if "." in key:
            idx_str = key.split(".", 1)[0]
            try:
                indices.add(int(idx_str))
            except ValueError:
                pass
    
    print(f"\nInferred indices: {sorted(indices)}")
    
    # Check first autoencoder if it exists
    if 0 in indices:
        w_enc = tensors.get("0.W_enc")
        w_dec = tensors.get("0.W_dec")
        if w_enc is not None and w_dec is not None:
            print(f"\nAutoencoder 0:")
            print(f"  W_enc: {w_enc.shape}")
            print(f"  W_dec: {w_dec.shape}")
            print(f"  Inferred d_in from W_enc[0]: {w_enc.shape[0]}")
            print(f"  Inferred d_sae from W_enc[1]: {w_enc.shape[1]}")
            print(f"  Expected W_dec shape: ({w_enc.shape[1]}, {w_enc.shape[0]})")
            print(f"  Actual W_dec shape: {w_dec.shape}")
            print(f"  Shapes match: {w_dec.shape == (w_enc.shape[1], w_enc.shape[0])}")

except Exception as e:
    print(f"Error: {e}")

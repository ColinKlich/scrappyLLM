#!/usr/bin/env python
"""
Convert PyTorch checkpoint (.pt) to safetensors format for easier loading and sharing.
Safetensors is a safer, faster format that's becoming the standard for model weights.
"""

import argparse
import torch
import os
from pathlib import Path


def convert_checkpoint_to_safetensors(checkpoint_path: str, output_path: str = None):
    """
    Convert a PyTorch checkpoint to safetensors format.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_path: Optional output path (defaults to same name with .safetensors)
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Error: safetensors package not installed.")
        print("Install with: pip install safetensors")
        return False

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Found 'model' key in checkpoint")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found 'state_dict' key in checkpoint")
    else:
        # Assume the checkpoint IS the state dict
        state_dict = checkpoint
        print("Using checkpoint as state_dict directly")

    # Print model info
    print(f"\nModel contains {len(state_dict)} tensors:")
    total_params = sum(t.numel() for t in state_dict.values())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Show a few layer names
    print("\nSample layers:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")
    if len(state_dict) > 5:
        print(f"  ... and {len(state_dict) - 5} more")

    # Convert tensors to contiguous format (required for safetensors)
    print("\nConverting tensors to contiguous format...")
    state_dict_contiguous = {}

    # Handle weight tying: if wte.weight and lm_head.weight share memory, clone one
    weight_tied = False
    if 'wte.weight' in state_dict and 'lm_head.weight' in state_dict:
        if state_dict['wte.weight'].data_ptr() == state_dict['lm_head.weight'].data_ptr():
            print("Detected weight tying between wte.weight and lm_head.weight")
            weight_tied = True

    for k, v in state_dict.items():
        # Clone lm_head.weight if it shares memory with wte.weight
        if weight_tied and k == 'lm_head.weight':
            state_dict_contiguous[k] = v.clone().contiguous()
            print(f"  Cloned {k} to break weight tying")
        else:
            state_dict_contiguous[k] = v.contiguous()

    # Determine output path
    if output_path is None:
        output_path = checkpoint_path.replace('.pt', '.safetensors')
    else:
        # If output is a directory, create filename in that directory
        if os.path.isdir(output_path):
            base_name = os.path.basename(checkpoint_path).replace('.pt', '.safetensors')
            output_path = os.path.join(output_path, base_name)

    # Save to safetensors
    print(f"\nSaving to {output_path}")
    save_file(state_dict_contiguous, output_path)

    # Save config separately as JSON
    config_output = output_path.replace('.safetensors', '_config.json')
    if 'config' in checkpoint:
        import json
        config = checkpoint['config']

        # Convert config to dict if it's an object
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config

        # Save config
        with open(config_output, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Saved config to {config_output}")

        # Print config
        print("\nModel configuration:")
        for key, value in config_dict.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")

    # Verify the conversion
    print("\nVerifying conversion...")
    from safetensors.torch import load_file
    loaded = load_file(output_path)

    # Check all keys match
    if set(loaded.keys()) == set(state_dict.keys()):
        print("✓ All keys match")
    else:
        print("✗ Key mismatch!")
        return False

    # Check a few tensors match
    mismatches = 0
    for key in list(state_dict.keys())[:10]:
        if not torch.allclose(loaded[key], state_dict[key], rtol=1e-5):
            mismatches += 1
            print(f"✗ Mismatch in {key}")

    if mismatches == 0:
        print("✓ All sampled tensors match")

    # File size comparison
    original_size = os.path.getsize(checkpoint_path)
    new_size = os.path.getsize(output_path)
    print(f"\nFile sizes:")
    print(f"  Original (.pt):      {original_size / 1e6:.2f} MB")
    print(f"  Safetensors:         {new_size / 1e6:.2f} MB")
    print(f"  Difference:          {(new_size - original_size) / 1e6:.2f} MB")

    print(f"\n✓ Conversion complete!")
    print(f"\nTo load the model:")
    print(f"  from safetensors.torch import load_file")
    print(f"  state_dict = load_file('{output_path}')")
    print(f"  model.load_state_dict(state_dict)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint to safetensors format"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to .pt checkpoint file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path (defaults to same name with .safetensors)'
    )

    args = parser.parse_args()

    # Check input exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # Convert
    success = convert_checkpoint_to_safetensors(args.checkpoint, args.output)

    if not success:
        print("\n✗ Conversion failed")
        exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Inspect tensors stored in a .safetensors file."""

import argparse
from pathlib import Path

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load an adapter_model.safetensors file and print stored tensors."
    )
    parser.add_argument("--f", type=Path, help="Path to .safetensors file")
    parser.add_argument(
        "--max-values",
        type=int,
        default=8,
        help="How many values to preview for each tensor (default: 8)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full tensor values instead of preview",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used to load tensors, e.g. cpu/cuda:0 (default: cpu)",
    )
    return parser.parse_args()


def preview_tensor(tensor, max_values: int) -> str:
    flat = tensor.reshape(-1)
    if flat.numel() <= max_values:
        return str(flat.tolist())
    preview = flat[:max_values].tolist()
    return f"{preview} ..."


def main() -> None:
    args = parse_args()

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "This script requires torch. Install it first, then rerun."
        ) from exc

    file_path = args.f

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    with safe_open(str(file_path), framework="pt", device=args.device) as f:
        metadata = f.metadata()
        keys = list(f.keys())

        print(f"File: {file_path}")
        print(f"Tensor count: {len(keys)}")
        print(f"Metadata: {metadata if metadata else '{}'}")
        print("-" * 80)

        for name in keys:
            tensor = f.get_tensor(name)
            print(f"name: {name}")
            print(f"shape: {tuple(tensor.shape)}")
            print(f"dtype: {tensor.dtype}")
            print(f"numel: {tensor.numel()}")

            if args.full:
                print("values:")
                print(tensor)
            else:
                print(f"values preview: {preview_tensor(tensor, args.max_values)}")

            print("-" * 80)


if __name__ == "__main__":
    main()

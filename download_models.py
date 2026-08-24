#!/usr/bin/env python3
"""
Download a HuggingFace model on CINECA Leonardo login node.

Usage:
    python download_model.py --model meta-llama/Llama-3.1-8B-Instruct --token hf_xxx --hf-home /leonardo_scratch/large/userexternal/{user}/.hf_cache
    python download_model.py --model mistralai/Mistral-7B-v0.1  # public models, no token needed
"""

import argparse
import os
from pathlib import Path
import dotenv

dotenv.load_dotenv()
user = os.environ.get("USER")


def main():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace model to $WORK cache"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo id, e.g. meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", None),
        help="HuggingFace token (required for gated models)",
    )
    parser.add_argument(
        "--hf-home",
        default=f"/leonardo_scratch/large/userexternal/{user}/.hf_cache",
        help=f"Cache directory (default: /leonardo_scratch/large/userexternal/{user}/.hf_cache)",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=["*.bin"],  # download safetensors only, skip pytorch bin
        help="File patterns to exclude from download (default: *.bin)",
    )
    args = parser.parse_args()

    # Set HF_HOME before importing huggingface_hub
    os.environ["HF_HOME"] = args.hf_home
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)
    print(f"HF_HOME = {args.hf_home}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not found. Install it with: pip install huggingface_hub")
        raise SystemExit(1)

    print(f"Downloading: {args.model}")
    if args.ignore_patterns:
        print(f"Ignoring patterns: {args.ignore_patterns}")

    path = snapshot_download(
        repo_id=args.model,
        token=args.token,
        ignore_patterns=args.ignore_patterns,
    )

    print(f"\nDownload complete!")
    print(f"  Path: {path}")
    print(f"\nTo use it in your code:")
    print(f'  os.environ["HF_HOME"] = "{args.hf_home}"')
    print(f'  resolve_model_path("{args.model}")')


if __name__ == "__main__":
    main()

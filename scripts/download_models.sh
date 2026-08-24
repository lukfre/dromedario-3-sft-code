#!/bin/bash
models=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    # "meta-llama/Llama-3.1-8B"
    "sapienzanlp/Minerva-350M-base-v1.0"
    "sapienzanlp/Minerva-1B-base-v1.0"
    "sapienzanlp/Minerva-3B-base-v1.0"
    # "sapienzanlp/Minerva-7B-base-v1.0"
)
for model in "${models[@]}"; do
    uv run download_models.py --model "$model" --hf-home "/leonardo_scratch/large/userexternal/$USER/.hf_cache"
done
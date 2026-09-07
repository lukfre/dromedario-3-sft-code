#!/bin/bash
models=(
    "meta-llama/Llama-3.1-8B"
    "sapienzanlp/Minerva-7B-base-v1.0"
)
for model in "${models[@]}"; do
    echo "Downloading $model"
    uv run download_models.py --model "$model" --hf-home "/leonardo_scratch/large/userexternal/$USER/.hf_cache"
done
#!/bin/bash
set -euo pipefail

# Check if the required models are present in the Hugging Face cache before starting training
export HF_HOME="/leonardo_scratch/large/userexternal/$USER/.hf_cache"
models=(
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "sapienzanlp/Minerva-7B-base-v1.0"
    "sapienzanlp/Minerva-350M-base-v1.0"
    "sapienzanlp/Minerva-1B-base-v1.0"
    "sapienzanlp/Minerva-3B-base-v1.0"
)
for model in "${models[@]}"; do
    if [ -d "$HF_HOME/hub/models--${model//\//--}" ]; then
        echo "> Model $model found in cache"
    else
        echo "> Model $model NOT found in cache!"; exit 1
    fi
done

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/tulu-it"
LOG_DIR="$PROJECT_DIR/code/logs"
SBATCH_SCRIPT="$PROJECT_DIR/code/scripts/multinode_sft.sbatch"

mkdir -p "$LOG_DIR"

configs=(
    # ===== LLAMA RUNS ON 7B MODELS =====
    # "$PROJECT_DIR/code/_configs/llama3___control.yaml"
    # "$PROJECT_DIR/code/_configs/llama3___add_50.yaml"
    # "$PROJECT_DIR/code/_configs/llama3___add_100.yaml"
    # "$PROJECT_DIR/code/_configs/llama3___sub_50.yaml"
    # "$PROJECT_DIR/code/_configs/llama3___sub_100.yaml"
    # ===== MINERVA RUNS ON 7B MODELS =====
    # # "$PROJECT_DIR/code/_configs/minerva___control.yaml"
    # "$PROJECT_DIR/code/_configs/minerva___add_50.yaml"
    # "$PROJECT_DIR/code/_configs/minerva___add_100.yaml"
    # "$PROJECT_DIR/code/_configs/minerva___sub_50.yaml"
    # "$PROJECT_DIR/code/_configs/minerva___sub_100.yaml"
    # ===== ADDITIONAL RUNS ON SMALLER MINERVA MODELS =====
    # "$PROJECT_DIR/code/_configs/minerva_0.4B___control.yaml"
    # "$PROJECT_DIR/code/_configs/minerva_0.4B___sub_100.yaml"
    # "$PROJECT_DIR/code/_configs/minerva_0.4B___add_100.yaml"
    # "$PROJECT_DIR/code/_configs/minerva_1B___control.yaml"
    # "$PROJECT_DIR/code/_configs/minerva_1B___sub_100.yaml"
    "$PROJECT_DIR/code/_configs/minerva_1B___add_100.yaml"
    "$PROJECT_DIR/code/_configs/minerva_3B___control.yaml"
    "$PROJECT_DIR/code/_configs/minerva_3B___sub_100.yaml"
    "$PROJECT_DIR/code/_configs/minerva_3B___add_100.yaml"
    # # ===== ADDITIONAL RUNS ON SMALLER LLAMA MODELS =====
    # "$PROJECT_DIR/code/_configs/llama3_3B___control.yaml"
    # "$PROJECT_DIR/code/_configs/llama3_3B___sub_100.yaml"
    # "$PROJECT_DIR/code/_configs/llama3_3B___add_100.yaml"
    # # "$PROJECT_DIR/code/_configs/llama3_1B___control.yaml"
    # # "$PROJECT_DIR/code/_configs/llama3_1B___sub_100.yaml"
    # "$PROJECT_DIR/code/_configs/llama3_1B___add_100.yaml"
)

for config in "${configs[@]}"; do
    base_name=$(basename "$config" .yaml)
    date_str=$(date +%Y-%m-%d_%H-%M-%S)
    out_log="$LOG_DIR/${base_name}/${date_str}_%j.out.log"
    err_log="$LOG_DIR/${base_name}/${date_str}_%j.err.log"
    mkdir -p "$(dirname "$out_log")"
    mkdir -p "$(dirname "$err_log")"

    # Determine number of nodes based on model size
    if [[ "$base_name" =~ 0.4B|1B ]]; then
        num_nodes=1
    elif [[ "$base_name" =~ 3B ]]; then
        num_nodes=2
    else
        num_nodes=4
    fi

    sbatch \
        --output="$out_log" \
        --error="$err_log" \
        --export=ALL,CMD="$config" \
        --nodes="$num_nodes" \
        "$SBATCH_SCRIPT"
    sleep 0.2  # Sleep briefly to avoid overwhelming the scheduler

done

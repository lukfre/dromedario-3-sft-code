#!/bin/bash
#SBATCH -A AIFAC_L04_006
#SBATCH --job-name=tulu-it_preprocessing
#SBATCH --output=/leonardo_scratch/large/userexternal/$USER/tulu-it/code/logs/prep_%j.out.log
#SBATCH --error=/leonardo_scratch/large/userexternal/$USER/tulu-it/code/logs/prep_%j.err.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --qos=normal
 
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================="
 
module purge
module load cuda/12.2
module load gcc
 
unset HF_HUB_OFFLINE
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/tulu-it/code/"
export VENV_DIR="$PROJECT_DIR/.venv"
cd $PROJECT_DIR
 
export TMPDIR="/leonardo_scratch/large/userexternal/$USER/tmp"
mkdir -p $TMPDIR
 
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/leonardo_scratch/large/userexternal/$USER/.hf_cache"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
mkdir -p $TORCH_EXTENSIONS_DIR
export TORCH_COMPILE_DISABLE=1
export PIP_CACHE_DIR="$TMPDIR/pip_cache"
export UV_CACHE_DIR="$TMPDIR/uv_cache"
export TIKTOKEN_CACHE_DIR="/leonardo_scratch/large/userexternal/$USER/tiktoken"
mkdir -p $PIP_CACHE_DIR $UV_CACHE_DIR $TIKTOKEN_CACHE_DIR
# export HF_TOKEN="_"
export PYTHONPATH=$PROJECT_DIR
 
mkdir -p $PROJECT_DIR/cineca/logs
 
source $VENV_DIR/bin/activate
 
export WANDB_MODE=offline

echo "========================================="
echo "Running: python prepare_dataset.py"
echo "========================================="

export VIRTUAL_ENV=$VENV_DIR
$VENV_DIR/bin/python prepare_dataset.py
# Environment setup

```bash
# Create virtual env for llama factory
uv venv llama_env --python 3.11
source llama_env/bin/activate

# Clone LLaMA Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# Use uv to sync dependencies
uv pip install -e .
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --reinstall # check your cuda version beforehand
uv pip install "transformers==4.57.3" # Note: we need to pin transformers to deal with with Minerva
uv pip install -r requirements/metrics.txt
uv pip install wandb 

pip download --no-deps deepspeed-kernels
pip download --no-deps deepspeed==0.14.4
# Install them
pip install deepspeed-kernels
pip install "deepspeed==0.14.4"
# srun into a compute node and run this to check
# srun -N 1 -A <project> --ntasks-per-node=1 --cpus-per-task=8 --partition=boost_usr_prod --gres=gpu:4 --gpus-per-task=4 --time 05:00:00 --pty /bin/bash

module load cuda
python - <<'PY'
import deepspeed
from deepspeed.ops.op_builder import CPUAdamBuilder
print("deepspeed:", deepspeed.__version__)
print("cpu_adam:", CPUAdamBuilder().is_compatible())
try:
    from deepspeed.ops.op_builder import FusedAdamBuilder
    print("fused_adam:", FusedAdamBuilder().is_compatible())
except Exception as e:
    print("fused_adam import failed:", e)
PY

# If fused_adam is compatible, then all good. 
# Otherwise, need to install deepspeed on a node with CUDA toolkit loaded.
```


# Logging inside the compute node

## Setup a public key on the login node
```bash
# Create a local SSH keypair on the cluster
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Append this newly created public key to your own authorized_keys file
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# Ensure strict, secure permissions (SSH will silently ignore keys if permissions are too wide)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

## Connect to the compute node (only when the job is running)

```bash
ssh lrdnXXXX # use squeue -u $USER to see your assigned compute nodes
```
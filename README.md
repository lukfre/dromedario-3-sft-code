# Dromedario 3

**Dromedario-3** is a large-scale Italian instruction-tuning dataset derived from the English Tülu 3 SFT mixture through a principled translation pipeline: instructions and responses are classified according to the [Natural Instructions](https://github.com/allenai/natural-instructions) taxonomy, classes are manually reviewed for translation safety, and items in validated classes are machine-translated into Italian. 

**Dromedario-3** is intended for supervised fine-tuning of Italian (and Italian/English bilingual) large language models, supporting the same broad range of tasks as Tülu 3 — open-ended dialogue, reasoning, coding, and knowledge-intensive instruction following — with Italian instruction-response pairs alongside the original English data.

## TL;DR:

Dromedario 3 is freely available on [HuggingFace](https://huggingface.co/datasets/sapienzanlp/dromedario3) and can be downloaded with the `datasets` library.

```python
from datasets import load_dataset

ds = load_dataset("sapienzanlp/Dromedario-3", split="train")

# Italian subset: rows with a usable translation
it_only = ds.filter(lambda x: x["messages_transl"] is not None)

# Optionally drop rows that were automatically corrected
it_safe = it_only.filter(lambda x: not x["flags"])
```

Every row carries both the original conversation and, where available, its Italian translation, so English-only, Italian-only, and mixed training sets can all be built from this single release.

---

# SFT training on CINECA
If you wish to train your model on Dromedario, you can follow these instructions.
The original experiments were carried out on the [Leonardo HPC](https://docs.hpc.cineca.it/hpc/leonardo.html) (CINECA), so they should scale well on any SLURM-based cluster.
We perfomed all training runs using 4 nodes in parallel, each equipped with 4 GPUs with 64GBs of VRAM. 
We handled the parallelism with [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/).

## Environment Setup

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

pip install deepspeed-kernels deepspeed==0.14.4

# Log into a compute node and run this to check
srun -N 1 -A YOUR_PROJECT --ntasks-per-node=1 --cpus-per-task=8 --partition=boost_usr_prod --gres=gpu:4 --gpus-per-task=4 --time 00:15:00 --pty /bin/bash

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
# Otherwise, you need to install deepspeed on a node with CUDA toolkit loaded.
```


## Logging inside the compute node

You may want to login into a compute node for debugging.
To do so, setup a public key on the login node.

```bash
# Create a local SSH keypair on the cluster
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Append this newly created public key to your own authorized_keys file
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# Ensure strict, secure permissions (SSH will silently ignore keys if permissions are too wide)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

Them, you can connect to the compute node (only when the job is running) with:

```bash
ssh lrdnXXXX # use squeue -u $USER to see your assigned compute nodes
```


## Training 

### Models

This script will download and convert the dataset into the shareGPT format. 
It will write on disk all recipes used in our paper
- `control`: identical with TULU3
- `sub_50`/`sub_100`: substitute half/all translatable items with their Italian translation
- `add_50/add_100`: add Italian translations of half/all translatable items 

```bash
uv run prepare_dataset.py # this generates a dataset_info.json file
mv dataset_info.json ./LLaMA-Factory/data/dataset_info.json
```

### Data

Leonardo compute nodes do not have access to internet, so model should be downloaded before starting any run. 
You can do so by running:

```bash
./scripts/download_models.sh    # write the HF model id in the script 
```

### Training run

We ran all trainings through the [LlamaFactory library](https://llamafactory.readthedocs.io/en/latest/).
Refer to the official documentation available [here](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html) for the setup.

Remember to change the log directories in the `.sbatch` scripts accordingly.

```bash
./scripts/submit_all_configs.sh     # sbatches all declared configs 
```
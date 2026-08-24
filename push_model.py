import os
from pathlib import Path
import dotenv

# from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from tap import Tap


class Args(Tap):
    dry: bool = False


base_path = Path("/leonardo_scratch/large/userexternal/$USER/tulu-it/saves")
model_paths = [
    "minerva-dromedario-control/checkpoint-2000",
]


def push_model_to_hf(path: Path, repo: str, private: bool):
    # Load your local model and tokenizer
    print("📥 Loading model and tokenizer into memory...")
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # Push directly to the hub
    print(f"🚀 Pushing model and tokenizer to Hugging Face Hub at '{repo}'...")
    model.push_to_hub(repo, private=private)
    tokenizer.push_to_hub(repo, private=private)

    print(f"🎉 Done! Model is live at https://huggingface.co/{repo}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    args = Args().parse_args()

    for model_path in model_paths:
        # model_name, other = model_path.split("|")
        # recipe = other.split("/")[0].replace("TULU-IT___", "tuluit-")
        model_name = model_path.split("/")[1]
        if model_name == "llama3.1":
            model_name = "llama"
        path = base_path / model_path
        repo_name = f"lukfre/{model_name}"
        print("-" * 50)
        print("📂:", path)
        print("🤗:", repo_name)
        if args.dry:
            continue
        push_model_to_hf(
            path=path, repo="lukfre/minerva-dromedario-control-2000", private=False
        )

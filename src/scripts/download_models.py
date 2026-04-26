from huggingface_hub import snapshot_download
import os

# 1. Configuration
HF_TOKEN = "huggingface token"
repo_ids = [
    # "black-forest-labs/FLUX.2-klein-4B",
    # "stabilityai/stable-diffusion-3-medium-diffusers",
    # "black-forest-labs/FLUX.2-dev",
    # "stabilityai/stable-diffusion-3.5-large",
    # "stabilityai/stable-diffusion-xl-base-1.0",
    # "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "second-state/stable-diffusion-3-medium-GGUF",
    "city96/stable-diffusion-3.5-large-gguf",
    "black-forest-labs/FLUX.2-dev-NVFP4",
    "black-forest-labs/FLUX.2-klein-9b-kv-fp8",
]

# 2. Set the destination path
# This will create a 'base_models' folder in your current directory,
# and a subfolder named after the model.
for id in repo_ids:
    download_dir = (
        f"[Root or somewhere else with symlink]/base_models/{id.replace('/', '_')}"
    )
    os.makedirs(download_dir, exist_ok=True)

    # 3. Download the repository
    snapshot_download(
        repo_id=id,
        local_dir=download_dir,
        # This filter prevents downloading redundant or legacy weight formats
        ignore_patterns=["*.bin", "*.ckpt", "*.h5", "*.msgpack", "*.bin.index.json"],
        token=HF_TOKEN,
    )

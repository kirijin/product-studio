"""First-run model downloader with progress indication."""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import torch


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "cache"


REQUIRED_MODELS = [
    {
        "id": "sdxl-base",
        "type": "diffusers",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "size_gb": 7.0,
        "description": "Image generation model (SDXL)",
    },
    {
        "id": "sdxl-controlnet-canny",
        "type": "diffusers",
        "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
        "size_gb": 1.6,
        "description": "ControlNet for guided composition",
    },
    {
        "id": "sdxl-vae",
        "type": "diffusers",
        "repo_id": "madebyollin/sdxl-vae-fp16-fix",
        "size_gb": 0.3,
        "description": "VAE for SDXL (quality fix)",
    },
]


def download_models(progress_callback=None):
    """Download all required models. Returns True if all OK.

    progress_callback: fn(current, total, message) called on each step.
    """
    total = len(REQUIRED_MODELS)
    all_ok = True

    os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)
    os.environ["HF_HOME"] = str(MODELS_DIR.parent / "hf_home")

    for i, model in enumerate(REQUIRED_MODELS):
        msg = f"[{i+1}/{total}] Downloading {model['description']} ({model['size_gb']}GB)..."
        if progress_callback:
            progress_callback(i, total, msg)
        else:
            print(msg)

        try:
            if model["type"] == "diffusers":
                snapshot_download(
                    repo_id=model["repo_id"],
                    local_dir=MODELS_DIR / model["repo_id"],
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            elif model["type"] == "auto":
                for f in model["files"]:
                    hf_hub_download(
                        repo_id=model["repo_id"],
                        filename=f,
                        local_dir=MODELS_DIR / model["repo_id"],
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
        except Exception as e:
            print(f"  FAILED: {e}")
            all_ok = False
            if progress_callback:
                progress_callback(i, total, f"FAILED: {model['description']} — {e}")

    if progress_callback:
        progress_callback(total, total, "Download complete!" if all_ok else "Some downloads failed — check console.")
    return all_ok


def check_models_exist():
    """Return True if all required models are already cached."""
    for model in REQUIRED_MODELS:
        model_dir = MODELS_DIR / model["repo_id"]
        if not model_dir.exists():
            return False
    return True


if __name__ == "__main__":
    print("Product Studio — Model Downloader")
    print("This will download ~9GB of models on first run.\n")
    ok = download_models()
    sys.exit(0 if ok else 1)

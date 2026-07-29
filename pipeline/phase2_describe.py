"""Phase 2: AI description generation via Ollama vision model."""

import os
import base64
import json
from pathlib import Path
import urllib.request
import urllib.error


OLLAMA_API = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5-vl:7b"
PROMPT = (
    "Describe this product in 5-15 words for a product label. "
    "Include brand (if visible), product type, and one key feature. "
    "Be concise and factual."
)


def _check_ollama():
    """Verify Ollama is reachable and has the model."""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            # Accept with or without :7b suffix
            found = any(DEFAULT_MODEL in m for m in models)
            return found, models
    except (urllib.error.URLError, ConnectionRefusedError, OSError) as e:
        return False, str(e)


def describe_image(image_path, model=DEFAULT_MODEL):
    """Send an image to Ollama vision model and return description."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "images": [b64],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 128,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_API,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
        return result.get("response", "").strip()


def run(manifest, progress_callback=None):
    """Generate AI descriptions for all images in the manifest.

    Returns updated manifest with descriptions set.
    """
    available, detail = _check_ollama()
    if not available:
        print(f"Ollama not available: {detail}")
        # Don't fail — user can type descriptions manually
        return manifest

    images = manifest.get("images", [])
    total = len(images)

    for idx, img_entry in enumerate(images):
        if img_entry.get("description_edited", False):
            # User already edited this one — skip
            continue

        image_path = img_entry["original_path"]
        if progress_callback:
            progress_callback(
                idx, total,
                f"Describing: {Path(image_path).name}"
            )

        try:
            description = describe_image(image_path)
            img_entry["description"] = description
            img_entry["description_source"] = "ai"
        except Exception as e:
            print(f"  Failed to describe {Path(image_path).name}: {e}")
            img_entry["description"] = img_entry.get("description", "")
            img_entry["description_source"] = "fallback"

    if progress_callback:
        progress_callback(total, total, "Descriptions generated. Review before continuing.")

    return manifest

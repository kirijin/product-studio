"""Phase 1: Background removal using rembg."""

import os
import traceback
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import numpy as np

try:
    from rembg import remove as rembg_remove, new_session
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False


def _validate_image(path):
    """Check image is valid and not excessively large. Returns (ok, msg, max_dim)."""
    max_pixels = 4096  # resize anything above this to prevent OOM
    try:
        with Image.open(path) as img:
            img.verify()  # quick integrity check
        # Re-open after verify (verify closes the file)
        with Image.open(path) as img:
            w, h = img.size
            if max(w, h) > max_pixels:
                return True, f"Large image ({w}x{h}) — will resize to {max_pixels}px", max_pixels
            return True, f"{w}x{h}", None
    except (UnidentifiedImageError, OSError) as e:
        return False, f"Corrupt or unsupported image: {e}", None


def _safe_resize(img, max_dim=4096):
    """Downscale if any dimension exceeds max_dim, maintaining aspect ratio."""
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img


def run(manifest, progress_callback=None):
    """Remove backgrounds from all images in the manifest.

    Returns updated manifest with cutout_path and mask_path set.
    Includes: image validation, size limiting, per-image error handling.
    """
    if not HAS_REMBG:
        raise RuntimeError(
            "rembg is not installed. Run: pip install rembg[gpu]"
        )

    # Error log path
    error_log = Path(manifest["session_dir"]) / "errors_phase1.log"

    session = new_session("briaai/RMBG-2.0")
    images = manifest.get("images", [])
    total = len(images)

    for idx, img_entry in enumerate(images):
        input_path = Path(img_entry["original_path"])
        if not input_path.exists():
            img_entry["error"] = f"File not found: {input_path}"
            continue

        if progress_callback:
            progress_callback(idx, total, f"Removing background: {input_path.name}")

        try:
            # Validate image
            ok, msg, max_dim = _validate_image(str(input_path))
            if not ok:
                raise ValueError(msg)

            img = Image.open(input_path).convert("RGBA")

            # Resize large images to prevent OOM
            if max_dim:
                img = _safe_resize(img, max_dim)
                if progress_callback:
                    progress_callback(idx, total, f"Resized {input_path.name} to {img.width}x{img.height}")

            # Check if already a cutout (existing alpha channel with transparent pixels)
            alpha = img.getchannel("A")
            alpha_array = np.array(alpha)
            has_transparency = np.any(alpha_array < 250)

            if has_transparency:
                # Already a cutout — just use as-is
                mask = Image.fromarray((alpha_array > 0).astype(np.uint8) * 255)
                cutout = img
            else:
                # Run rembg
                output = rembg_remove(img, session=session)
                cutout = output.convert("RGBA")
                mask = cutout.getchannel("A")

            # Save cutout
            cutout_dir = Path(manifest["session_dir"]) / "cutouts"
            cutout_dir.mkdir(parents=True, exist_ok=True)
            cutout_path = cutout_dir / f"{input_path.stem}_cutout.png"
            cutout.save(cutout_path)
            img_entry["cutout_path"] = str(cutout_path)

            # Save mask
            mask_dir = Path(manifest["session_dir"]) / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{input_path.stem}_mask.png"
            mask.save(mask_path)
            img_entry["mask_path"] = str(mask_path)

            img_entry["original_size"] = f"{img.width}x{img.height}"

        except Exception as e:
            err_msg = f"{input_path.name}: {e}"
            print(f"  FAILED {err_msg}")
            img_entry["error"] = err_msg
            # Log full traceback
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"[{idx+1}/{total}] {err_msg}\n{traceback.format_exc()}\n")
            continue

    if progress_callback:
        ok_count = sum(1 for i in images if i.get("cutout_path"))
        fail_count = sum(1 for i in images if i.get("error"))
        status = f"Background removal: {ok_count} OK"
        if fail_count:
            status += f", {fail_count} failed — see {error_log.name}"
        progress_callback(total, total, status)

    # Clean up rembg session
    del session

    return manifest

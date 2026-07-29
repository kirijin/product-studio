"""Phase 3: Background generation, compositing, text overlay, and final output.

This is the GPU-heavy phase. Loads SDXL once, processes all images, unloads.
"""

import hashlib
import math
import os
import gc
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageChops
import numpy as np

from . import smart_placement
from . import styles as style_module
from . import bg_prompts

# ── SDXL / CUDA (lazy-loaded) ──────────────────────────────────────────
_pipe = None
_controlnet = None
_canny_processor = None
_depth_processor = None


def _unload_models():
    global _pipe, _controlnet, _canny_processor, _depth_processor
    _pipe = None
    _controlnet = None
    _canny_processor = None
    _depth_processor = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def _load_pipeline():
    global _pipe, _controlnet, _canny_processor, _depth_processor

    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import (
        StableDiffusionXLControlNetPipeline,
        ControlNetModel,
        AutoencoderKL,
        EulerDiscreteScheduler,
    )
    from controlnet_aux import CannyDetector, DepthDetector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ── VRAM check → decide offload strategy ──
    vram_gb = 0
    use_cpu_offload = False
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        if vram_gb < 11:
            use_cpu_offload = True
            print(f"  VRAM={vram_gb:.1f}GB — enabling CPU offload to prevent OOM")

    print(f"  Loading SDXL (device={device}, dtype={dtype}, offload={use_cpu_offload})...")

    # ControlNet for composition guidance
    _controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # VAE with fp16 fix for quality
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=dtype,
    )

    # Main pipeline
    _pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=_controlnet,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    )
    _pipe.scheduler = EulerDiscreteScheduler.from_config(_pipe.scheduler.config)

    # Memory optimizations
    if device == "cuda":
        _pipe.enable_xformers_memory_efficient_attention()
        if use_cpu_offload:
            _pipe.enable_model_cpu_offload()
        else:
            _pipe = _pipe.to(device)
            print(f"  Full GPU mode ({vram_gb:.0f}GB VRAM)")
    else:
        _pipe = _pipe.to(device)

    # Processors for ControlNet condition images
    _canny_processor = CannyDetector()
    _depth_processor = DepthDetector.from_pretrained("LiheYoung/depth-anything-base-hf")

    return _pipe


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_seed(filename: str, seed_mode: str) -> int:
    """Deterministic seed from filename, or fixed seed."""
    if seed_mode == "fixed":
        return 42
    # Deterministic: hash filename
    h = hashlib.sha256(filename.encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def _composite_product(background, cutout, mask, feather_px=8):
    """Composite cutout onto background with feathered edges.

    All images are RGBA PIL Images of the same size.
    Returns composited RGBA image.
    """
    bg = background.convert("RGBA")
    ct = cutout.convert("RGBA")

    # Feather the mask edges
    mask_img = mask.convert("L")
    if feather_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_px))

    # Composite using the feathered mask
    result = bg.copy()
    result.paste(ct, (0, 0), mask_img)
    return result


def _render_label(image, text, style_name, position, mask_for_placement):
    """Render text label on the image using the given style.

    Returns image with label rendered.
    """
    style = style_module.LABEL_STYLES.get(style_name, style_module.LABEL_STYLES["badge"])

    # Font
    font_path = style_module.FONT_PATHS.get(style["font_family"])
    if font_path:
        font_path_resolved = Path(__file__).resolve().parent.parent / font_path
        if font_path_resolved.exists():
            font = ImageFont.truetype(str(font_path_resolved), style["font_size"])
        else:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # Measure text
    draw = ImageDraw_dummy()
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    total_w = text_w + style["padding_x"] * 2
    total_h = text_h + style["padding_y"] * 2

    # Find placement
    pos = position if position != "auto" else style["default_position"]
    x, y, _ = smart_placement.find_safe_placement(
        mask_for_placement,
        text_width=total_w,
        text_height=total_h,
        position_preference=pos,
        image_width=image.width,
        image_height=image.height,
    )

    # Create label surface
    label = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label)

    if style["bg_color"] is not None:
        # Background pill/rect
        bg = style["bg_color"]
        if style["border_radius"] > 0:
            label_draw.rounded_rectangle(
                [(0, 0), (total_w - 1, total_h - 1)],
                radius=style["border_radius"],
                fill=bg,
            )
        else:
            label_draw.rectangle(
                [(0, 0), (total_w - 1, total_h - 1)],
                fill=bg,
            )

    # Shadow
    if style.get("shadow", False) and style["bg_color"] is not None:
        shadow = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        sx, sy = style["shadow_offset"]
        if style["border_radius"] > 0:
            shadow_draw.rounded_rectangle(
                [(sx, sy), (total_w - 1 + sx, total_h - 1 + sy)],
                radius=style["border_radius"],
                fill=style["shadow_color"],
            )
        else:
            shadow_draw.rectangle(
                [(sx, sy), (total_w - 1 + sx, total_h - 1 + sy)],
                fill=style["shadow_color"],
            )
        label = Image.alpha_composite(shadow, label)

    # Text
    text_color = style["text_color"]
    label_draw = ImageDraw.Draw(label)
    text_x = style["padding_x"]
    text_y = style["padding_y"]
    label_draw.text((text_x, text_y), text, fill=text_color, font=font)

    # Composite onto image
    result = image.copy().convert("RGBA")
    result.paste(label, (x, y), label)
    return result


class ImageDraw_dummy:
    """Minimal draw object for measurement-only use."""
    pass


# ── Main ────────────────────────────────────────────────────────────────


def run(manifest, progress_callback=None):
    """Generate backgrounds, composite products, render labels.

    Returns updated manifest with output_path set.
    """
    import torch

    config = manifest.get("config", {})
    bg_style = config.get("bg_style", "studio")
    bg_prompt_custom = config.get("bg_prompt", "")
    label_style = config.get("label_style", "badge")
    label_position = config.get("label_position", "auto")
    denoise_strength = config.get("denoise", 0.0)
    saturation_factor = config.get("saturation", 1.0)
    upscale = config.get("upscale", "none")
    seed_mode = config.get("seed_mode", "deterministic")

    # Resolve background prompt
    preset = bg_prompts.BACKGROUND_PRESETS.get(bg_style, bg_prompts.BACKGROUND_PRESETS["studio"])
    full_prompt = bg_prompt_custom if bg_prompt_custom else preset["prompt"]
    negative_prompt = preset["negative"]
    controlnet_type = preset.get("controlnet", "canny")

    # Determine output dimensions
    output_size = config.get("output_size", "1024")

    # Load pipeline
    pipe = _load_pipeline()

    images = manifest.get("images", [])
    total = len(images)
    session_dir = Path(manifest["session_dir"])
    output_dir = session_dir.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log = session_dir / "errors_phase3.log"

    # WDDM guard: reduce steps if VRAM < 10GB (Windows GPU watchdog)
    inference_steps = 25 if not use_cpu_offload else 20
    pipe = _load_pipeline()

    for idx, img_entry in enumerate(images):
        cutout_path = Path(img_entry.get("cutout_path", ""))
        mask_path = Path(img_entry.get("mask_path", ""))
        img_name = cutout_path.stem.replace("_cutout", "") if cutout_path.suffix else img_entry.get("original_path", "unknown")

        if "error" in img_entry:
            continue

        if progress_callback:
            progress_callback(idx, total, f"Generating: {img_name}")

        try:
            cutout = Image.open(cutout_path).convert("RGBA")
            mask = Image.open(mask_path).convert("L")

            # Resize to working resolution
            target_size = int(output_size)
            cutout = _resize_with_pad(cutout, target_size)
            mask = _resize_with_pad(mask, target_size, fill=0)

            # ── Generate background ──
            seed = _make_seed(img_name, seed_mode)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            # Build ControlNet condition image
            if controlnet_type == "canny":
                condition = _canny_processor(cutout, low_threshold=100, high_threshold=200)
            elif controlnet_type == "depth":
                condition = _depth_processor(cutout)
            else:
                condition = Image.new("L", (target_size, target_size), 128)

            # Ensure condition is RGB
            condition = condition.convert("RGB") if condition.mode != "RGB" else condition

            # Generate
            result = pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=condition,
                num_inference_steps=25,
                guidance_scale=7.5,
                generator=generator,
                height=target_size,
                width=target_size,
                controlnet_conditioning_scale=0.8,
            ).images[0]

            background = result.convert("RGBA")

            # ── Composite product onto background ──
            composited = _composite_product(background, cutout, mask, feather_px=6)

            # ── Post-process: denoise, saturate ──
            composited = _apply_enhancements(
                composited, denoise=denoise_strength, saturation=saturation_factor
            )

            # ── Render text label ──
            label_text = img_entry.get("description", "")
            if label_text:
                # Use mask for smart placement (inverted: product = white)
                placement_mask = mask
                composited = _render_label(
                    composited, label_text, label_style, label_position, placement_mask
                )

            # ── Upscale if requested ──
            if upscale == "2x":
                composited = composited.resize(
                    (composited.width * 2, composited.height * 2), Image.LANCZOS
                )
            elif upscale == "4x":
                composited = composited.resize(
                    (composited.width * 4, composited.height * 4), Image.LANCZOS
                )

            # ── Save ──
            output_path = output_dir / f"{img_name}_studio.png"
            composited.convert("RGB").save(output_path, quality=95)
            img_entry["output_path"] = str(output_path)

        except torch.cuda.OutOfMemoryError:
            err_msg = f"CUDA OOM on {img_name} — enabling CPU offload"
            print(f"  ! {err_msg}")
            img_entry["error"] = err_msg
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"[{idx+1}/{total}] {err_msg}\n")
            # Emergency: enable CPU offload and skip remaining
            _pipe.enable_model_cpu_offload()
            continue

        except Exception as e:
            err_msg = f"{img_name}: {e}"
            print(f"  FAILED {err_msg}")
            img_entry["error"] = err_msg
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"[{idx+1}/{total}] {err_msg}\n")
            continue

    # Final output
    manifest["output_dir"] = str(output_dir)

    if progress_callback:
        n_ok = sum(1 for i in images if i.get("output_path"))
        n_fail = sum(1 for i in images if i.get("error"))
        status = f"Done: {n_ok} succeeded"
        if n_fail:
            status += f", {n_fail} failed"
        progress_callback(total, total, status)

    return manifest


def _resize_with_pad(img, target_size, fill=255):
    """Resize image to target_size x target_size with padding.

    Maintains aspect ratio; pads shorter dimension with `fill`.
    """
    img = img.convert("RGBA") if img.mode != "RGBA" else img
    ratio = target_size / max(img.width, img.height)
    new_w = int(img.width * ratio)
    new_h = int(img.height * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    padded = Image.new("RGBA", (target_size, target_size), (fill, fill, fill, 0))
    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2
    padded.paste(img, (x, y), img if img.mode == "RGBA" else None)
    return padded


def _apply_enhancements(img, denoise=0.0, saturation=1.0):
    """Apply denoising and saturation adjustments."""
    result = img.convert("RGB")

    if saturation != 1.0:
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(saturation)

    if denoise > 0:
        import cv2
        arr = np.array(result)
        h = int(denoise * 10)  # denoise: 0..1 map to 0..10
        h = max(1, h) if h > 0 else 0
        if h > 0:
            arr = cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)
            result = Image.fromarray(arr)

    return result.convert("RGBA")

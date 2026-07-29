"""Product Studio — Main Gradio Application.

Two-screen UI for layman users:
1. Upload product images, generate/edit text descriptions
2. Set style, process all images, download results
"""

import os
import json
import uuid
import shutil
import zipfile
import tempfile
from pathlib import Path
import threading

import gradio as gr
from PIL import Image

from pipeline import phase1_remove_bg, phase2_describe, phase3_render
from pipeline.styles import LABEL_STYLES, POSITION_OPTIONS
from pipeline.bg_prompts import BACKGROUND_PRESETS
from models import download_fonts


# ── Configuration ──────────────────────────────────────────────────────

APP_TITLE = "Product Studio"
SESSION_DIR = Path(__file__).resolve().parent / "state"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SESSION_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ── State management ───────────────────────────────────────────────────


def new_session():
    """Create a new session with a unique ID."""
    session_id = uuid.uuid4().hex[:12]
    session_dir = SESSION_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "images": [],
        "config": {
            "bg_style": "studio",
            "bg_prompt": "",
            "label_style": "badge",
            "label_position": "auto",
            "denoise": 0,
            "saturation": 1.0,
            "upscale": "none",
            "output_size": "1024",
            "seed_mode": "deterministic",
        },
    }


def save_manifest(manifest):
    """Persist manifest to disk."""
    path = Path(manifest["session_dir"]) / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def load_manifest(session_id):
    """Load manifest from disk."""
    path = SESSION_DIR / session_id / "manifest.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _get_thumbnail(path, size=200):
    """Create a thumbnail for gallery display."""
    if not path or not Path(path).exists():
        return None
    img = Image.open(path)
    img.thumbnail((size, size))
    # Return as numpy for Gradio
    import numpy as np
    return np.array(img)


# ── Processing functions (called by Gradio) ───────────────────────────


def on_upload(files, manifest_state):
    """Handle file upload — copy to session dir, create entries."""
    if not files:
        return manifest_state, [], gr.update(visible=True)

    manifest = manifest_state or new_session()
    session_dir = Path(manifest["session_dir"])
    inputs_dir = session_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    existing_names = {img["filename"] for img in manifest["images"]}
    new_entries = []

    for file_obj in files:
        fname = Path(file_obj.name).name
        if fname in existing_names:
            continue
        # Copy to session inputs
        dest = inputs_dir / fname
        shutil.copy2(file_obj.name, dest)
        entry = {
            "filename": fname,
            "original_path": str(dest),
            "cutout_path": None,
            "mask_path": None,
            "description": "",
            "description_source": "manual",
            "description_edited": False,
            "output_path": None,
            "error": None,
        }
        new_entries.append(entry)
        existing_names.add(fname)

    manifest["images"].extend(new_entries)
    save_manifest(manifest)

    # Build gallery data
    gallery = _build_gallery_data(manifest["images"])
    return manifest, gallery, gr.update(visible=True)


def _build_gallery_data(images):
    """Build list of (thumbnail, label) tuples for Gradio gallery."""
    result = []
    for img in images:
        thumb = _get_thumbnail(img.get("original_path"))
        label = img.get("description", "") or img.get("filename", "")
        result.append((thumb, label))
    return result


def on_describe_click(manifest_state, progress=gr.Progress()):
    """Run Phase 2: AI description generation."""
    if not manifest_state or not manifest_state.get("images"):
        return manifest_state, _build_gallery_data([])

    manifest = manifest_state

    def progress_cb(current, total, msg):
        progress((current, total), desc=msg)

    try:
        manifest = phase2_describe.run(manifest, progress_callback=progress_cb)
        save_manifest(manifest)
    except Exception as e:
        raise gr.Error(f"Description generation failed: {e}")

    gallery = _build_gallery_data(manifest["images"])
    return manifest, gallery


def on_descriptions_edit(manifest_state, *text_values):
    """Update descriptions from text inputs."""
    if not manifest_state:
        return manifest_state, []
    manifest = manifest_state
    for i, text in enumerate(text_values):
        if i < len(manifest["images"]):
            manifest["images"][i]["description"] = text
            manifest["images"][i]["description_edited"] = True
    save_manifest(manifest)
    return manifest, _build_gallery_data(manifest["images"])


def on_process_click(
    manifest_state,
    bg_style,
    bg_prompt,
    label_style,
    label_position,
    denoise,
    saturation,
    upscale,
    progress=gr.Progress(),
):
    """Run Phase 1 + Phase 3: remove background, generate, composite, label."""
    if not manifest_state or not manifest_state.get("images"):
        raise gr.Error("No images to process. Upload images first.")

    manifest = manifest_state
    images = manifest["images"]

    # Update config
    manifest["config"].update({
        "bg_style": bg_style,
        "bg_prompt": bg_prompt,
        "label_style": label_style,
        "label_position": label_position,
        "denoise": denoise,
        "saturation": saturation,
        "upscale": upscale,
    })

    # Phase 1: Background removal
    def cb_p1(c, t, m):
        progress((c, t * 2), desc=m)

    try:
        progress(0, desc="Starting background removal...")
        manifest = phase1_remove_bg.run(manifest, progress_callback=cb_p1)
        save_manifest(manifest)
    except Exception as e:
        raise gr.Error(f"Background removal failed: {e}")

    # Phase 3: Render
    def cb_p3(c, t, m):
        progress((len(images) + c, len(images) * 2), desc=m)

    try:
        manifest = phase3_render.run(manifest, progress_callback=cb_p3)
        save_manifest(manifest)
    except Exception as e:
        raise gr.Error(f"Rendering failed: {e}")

    # Build output gallery
    output_gallery = _build_output_gallery(manifest["images"])
    success_count = sum(1 for i in manifest["images"] if i.get("output_path"))
    fail_count = sum(1 for i in manifest["images"] if i.get("error"))

    msg = f"Complete! {success_count} images processed"
    if fail_count:
        msg += f", {fail_count} failed"

    return manifest, output_gallery, gr.update(visible=True, value=msg)


def _build_output_gallery(images):
    """Build gallery of output images."""
    result = []
    for img in images:
        path = img.get("output_path")
        if path and Path(path).exists():
            thumb = _get_thumbnail(path)
            label = img.get("description", "")
            result.append((thumb, label))
    return result


def on_download_zip(manifest_state):
    """Create a ZIP of all output images and return path."""
    if not manifest_state:
        raise gr.Error("Nothing to download.")

    images = manifest_state.get("images", [])
    output_paths = [
        img["output_path"] for img in images
        if img.get("output_path") and Path(img["output_path"]).exists()
    ]

    if not output_paths:
        raise gr.Error("No output images found. Process images first.")

    zip_path = str(Path(manifest_state["session_dir"]) / "product_studio_output.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in output_paths:
            zf.write(p, Path(p).name)

    return zip_path


def on_new_session():
    """Reset to a fresh session."""
    manifest = new_session()
    return manifest, [], gr.update(visible=False), gr.update(visible=False), gr.update(value="")


# ── UI Building ────────────────────────────────────────────────────────


def build_ui():
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as app:
        # State
        manifest_state = gr.State(new_session)

        # ── Screens ──

        # --- Screen 1: Upload & Describe ---
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown("Upload product photos, describe them, then generate polished studio images with labels.")

        with gr.Column() as upload_screen:
            file_input = gr.File(
                label="Drop product images here",
                file_count="multiple",
                file_types=["image", ".jpg", ".jpeg", ".png", ".webp"],
                type="filepath",
            )

            with gr.Row():
                upload_count = gr.Markdown("No images uploaded yet")

            describe_btn = gr.Button("Auto-generate descriptions (AI)", variant="secondary", visible=False)
            # (Gallery preview is shown in the description table thumbnails)

            # Editable text fields for descriptions
            # Descriptions table — editable
            descriptions_df = gr.Dataframe(
                headers=["Filename", "Description (click to edit)"],
                datatype=["str", "str"],
                col_count=(2, "fixed"),
                interactive=True,
                visible=False,
            )

            with gr.Row():
                apply_desc_btn = gr.Button("Save description edits", variant="primary", size="sm", visible=False)
                next_btn = gr.Button("Next: Set Style →", variant="primary", visible=False, size="lg")

        # --- Screen 2: Style & Generate ---
        with gr.Column(visible=False) as style_screen:
            gr.Markdown("## Style Settings")

            with gr.Row():
                with gr.Column():
                    bg_style = gr.Dropdown(
                        label="Background style",
                        choices=[(v["name"], k) for k, v in BACKGROUND_PRESETS.items()],
                        value="studio",
                    )
                    bg_prompt = gr.Textbox(
                        label="Custom background prompt (optional)",
                        placeholder="e.g., 'a wooden table, soft natural light'",
                        lines=2,
                    )
                with gr.Column():
                    label_style = gr.Dropdown(
                        label="Label style",
                        choices=[(v["name"], k) for k, v in LABEL_STYLES.items()],
                        value="badge",
                    )
                    label_position = gr.Dropdown(
                        label="Label position",
                        choices=[(p.replace("-", " ").title(), p) for p in POSITION_OPTIONS],
                        value="auto",
                    )

            with gr.Row():
                denoise = gr.Slider(
                    label="Denoise",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=0,
                )
                saturation = gr.Slider(
                    label="Saturation",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                )
                upscale = gr.Dropdown(
                    label="Upscale output",
                    choices=["none", "2x", "4x"],
                    value="none",
                )

            process_btn = gr.Button("Process All Images", variant="primary", size="lg")

            status_text = gr.Markdown(visible=False)

            gallery_output = gr.Gallery(
                label="Output Images",
                columns=4,
                height=400,
                object_fit="contain",
                show_label=True,
            )

            download_btn = gr.Button("Download ZIP", variant="secondary", visible=False)

        # Bottom nav
        with gr.Row():
            new_btn = gr.Button("Start New Session", variant="stop", size="sm")

        # ── Event wiring ────────────────────────────────────────────────

        # Upload
        def on_upload_with_df(files, manifest_state):
            manifest, _, _ = on_upload(files, manifest_state)
            # Build dataframe from descriptions
            rows = []
            for img in manifest.get("images", []):
                rows.append([img.get("filename", ""), img.get("description", "")])
            return manifest, rows, gr.update(visible=True), gr.update(visible=True)

        file_input.change(
            fn=on_upload_with_df,
            inputs=[file_input, manifest_state],
            outputs=[manifest_state, descriptions_df, describe_btn, descriptions_df],
        )
        file_input.change(
            fn=lambda files: f"**{len(files) if files else 0} images uploaded**",
            inputs=[file_input],
            outputs=[upload_count],
        )

        # Describe (AI)
        def on_describe_and_refresh(manifest_state, progress=gr.Progress()):
            manifest = on_describe_click(manifest_state, progress=progress)
            rows = []
            for img in manifest.get("images", []):
                rows.append([img.get("filename", ""), img.get("description", "")])
            return manifest, rows

        describe_btn.click(
            fn=on_describe_and_refresh,
            inputs=[manifest_state],
            outputs=[manifest_state, descriptions_df],
        )

        # Apply description edits from dataframe
        def apply_desc_edits(manifest_state, df_data):
            if not manifest_state or not df_data:
                return manifest_state
            manifest = manifest_state
            for i, row in enumerate(df_data):
                if i < len(manifest["images"]):
                    # row[0] = filename, row[1] = description
                    manifest["images"][i]["description"] = str(row[1])
                    manifest["images"][i]["description_edited"] = True
            save_manifest(manifest)
            return manifest

        apply_desc_btn.click(
            fn=apply_desc_edits,
            inputs=[manifest_state, descriptions_df],
            outputs=[manifest_state],
        )
        next_btn.click(
            fn=lambda ms: (gr.update(visible=False), gr.update(visible=True)),
            inputs=[],
            outputs=[upload_screen, style_screen],
        )

        # Back navigation
        def go_back():
            return gr.update(visible=True), gr.update(visible=False)

        # Process
        process_btn.click(
            fn=on_process_click,
            inputs=[
                manifest_state,
                bg_style,
                bg_prompt,
                label_style,
                label_position,
                denoise,
                saturation,
                upscale,
            ],
            outputs=[manifest_state, gallery_output, status_text],
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[download_btn],
        )

        # Download
        download_btn.click(
            fn=on_download_zip,
            inputs=[manifest_state],
            outputs=[gr.File(label="Download ZIP")],
        )

        # New session
        new_btn.click(
            fn=on_new_session,
            inputs=[],
            outputs=[manifest_state, descriptions_df, describe_btn, style_screen, status_text],
        ).then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(value=None)),
            inputs=[],
            outputs=[upload_screen, style_screen, descriptions_df],
        )

        # Example images
        gr.Examples(
            examples=[],
            inputs=[],
            label="No examples yet — upload your product photos above.",
        )

    return app


# ── Entry point ────────────────────────────────────────────────────────


if __name__ == "__main__":
    import webbrowser

    print("Starting Product Studio...")
    print("Open http://127.0.0.1:7860 in your browser.\n")

    # Try to open browser automatically
    threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:7860")).start()

    # Download fonts if missing
    if not download_fonts.check_fonts_exist():
        print("Downloading fonts for label rendering...")
        download_fonts.download_fonts()
    else:
        print("Fonts found.")

    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        quiet=False,
    )

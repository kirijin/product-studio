"""Quick pipeline smoke test v3 — correct manifest format."""
import sys, os, tempfile, json
os.environ["HF_HOME"] = r"C:\Users\user\Desktop\product-studio-master\models\cache\hf_home"
os.environ["HF_HUB_CACHE"] = r"C:\Users\user\Desktop\product-studio-master\models\cache"
sys.path.insert(0, r"C:\Users\user\Desktop\product-studio-master")

import PIL.Image

print("1. rembg remove...")
import rembg
img = PIL.Image.new("RGB", (128, 128), color=(255, 0, 0))
result = rembg.remove(img)
print(f"   OK: {result.size} {result.mode}")

print("\n2. Loading SDXL pipeline...")
from pipeline.phase3_render import _load_pipeline
pipe = _load_pipeline()
print(f"   OK: {type(pipe).__name__}")

print("\n3. Running full pipeline (phase1 + phase3)...")
from pipeline.phase1_remove_bg import run as bg_remove_run
from pipeline.phase3_render import run as render_run

session_dir = tempfile.mkdtemp(prefix="ps_test_")
product_path = os.path.join(session_dir, "product.png")
img.save(product_path)

manifest = {
    "product_image_path": product_path,
    "session_dir": session_dir,
    "output_dir": session_dir,
}

# Phase 1: remove background
bg_remove_run({"product_image_path": product_path, "output_dir": session_dir})
print("   Phase1 (bg removal): done")

# Phase 3: render
render_run(manifest)
print("   Phase3 (SDXL render): done")

import glob
out_files = glob.glob(os.path.join(session_dir, "*"))
print(f"   Output files: {[os.path.basename(f) for f in out_files]}")

print("\n=== ALL TESTS PASSED ===")

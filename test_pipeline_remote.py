"""Quick pipeline smoke test v2."""
import sys, os
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

print("\n3. Running pipeline (full render)...")
from pipeline.phase3_render import run as render_run
import tempfile
tmp = tempfile.mktemp(suffix=".png")
img.save(tmp)
manifest = {"product_image_path": tmp, "output_dir": os.path.dirname(tmp)}
render_run(manifest)
print(f"   OK")

print("\n=== ALL TESTS PASSED ===")

"""First-run environment check. Runs before Gradio starts.

Verifies: Python version, CUDA availability, VRAM, disk space, Ollama status,
critical Python packages. Prints a summary and returns exit code.
"""

import sys
import os
import shutil
import subprocess
import json
import urllib.request
import urllib.error
from pathlib import Path


# ANSI on Windows (most terminals support it now)
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

checks = []
failures = []
warnings = []


def ok(msg):
    checks.append(("PASS", msg))
    print(f"  {GREEN}[OK]{RESET} {msg}")


def warn(msg):
    checks.append(("WARN", msg))
    warnings.append(msg)
    print(f"  {YELLOW}[!]{RESET} {msg}")


def fail(msg):
    checks.append(("FAIL", msg))
    failures.append(msg)
    print(f"  {RED}[X]{RESET} {msg}")


def check_python():
    v = sys.version_info
    if v.major == 3 and v.minor >= 10:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor}.{v.micro} — need 3.10+")


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or "unknown"
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_total / 1e9
            ok(f"CUDA {cuda_ver} — {device_name} ({vram_gb:.1f} GB VRAM, {device_count} device(s))")

            # Warn if low VRAM
            if vram_gb < 10:
                warn(f"VRAM < 10GB ({vram_gb:.1f}) — will enable CPU offload for SDXL")
            return vram_gb
        else:
            warn("CUDA not available — will run on CPU (very slow, 30-60s per image)")
            return 0
    except ImportError:
        fail("PyTorch not installed — run install.bat")
        return 0
    except Exception as e:
        warn(f"CUDA check failed: {e}")
        return 0


def check_disk_space(min_gb=15):
    """Check free space in project folder drive."""
    project_dir = Path(__file__).resolve().parent
    try:
        free_bytes = shutil.disk_usage(project_dir).free
        free_gb = free_bytes / 1e9
        if free_gb >= min_gb:
            ok(f"Disk space: {free_gb:.1f} GB free")
        else:
            fail(f"Low disk space: {free_gb:.1f} GB free (need at least {min_gb} GB)")
    except Exception as e:
        warn(f"Could not check disk space: {e}")


def check_ollama():
    """Check if Ollama is running and has the required model."""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if any("qwen2.5-vl" in m for m in models):
                ok("Ollama running — qwen2.5-vl model available")
            else:
                warn("Ollama running but qwen2.5-vl:7b not pulled yet (will be auto-pulled on describe)")
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        warn("Ollama not running — AI descriptions will be unavailable (manual text entry only)")


def check_critical_packages():
    """Check importable packages."""
    required = {
        "PIL": "Pillow",
        "rembg": "rembg",
        "torch": "PyTorch",
        "gradio": "Gradio",
        "diffusers": "diffusers",
        "cv2": "opencv-python",
    }
    for mod_name, pkg_name in required.items():
        try:
            __import__(mod_name)
            ok(f"{pkg_name} installed")
        except ImportError:
            fail(f"{pkg_name} not installed — run install.bat again")


def check_gpu_driver():
    """Check NVIDIA driver version via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            driver_ver = result.stdout.strip()
            ok(f"NVIDIA driver: {driver_ver}")
        else:
            warn("nvidia-smi failed — GPU driver may be missing or outdated")
    except FileNotFoundError:
        warn("nvidia-smi not found — NVIDIA driver may not be installed")
    except Exception as e:
        warn(f"GPU driver check failed: {e}")


def check_windows_defender():
    """Warn about potential antivirus interference."""
    # Can't reliably check without admin, but warn proactively
    warn("Windows Defender / Antivirus may slow first run. Add exceptions for: product-studio/")


def run_all():
    print(f"\n{BOLD}Product Studio — Environment Check{RESET}")
    print("=" * 50)

    check_python()
    vram_gb = check_cuda()

    if vram_gb > 0:
        check_gpu_driver()

    check_disk_space()
    check_ollama()
    check_critical_packages()
    check_windows_defender()

    print("=" * 50)
    passed = sum(1 for s in checks if s[0] == "PASS")
    total = len(checks)
    print(f"  {BOLD}Summary:{RESET} {passed}/{total} passed")

    if warnings:
        print(f"  {YELLOW}{len(warnings)} warning(s):{RESET}")
        for w in warnings[:3]:
            print(f"    {YELLOW}[!]{RESET} {w}")
        if len(warnings) > 3:
            print(f"    ... and {len(warnings) - 3} more")

    if failures:
        print(f"  {RED}{len(failures)} failure(s) — fix before running:{RESET}")
        for f in failures:
            print(f"    {RED}[X]{RESET} {f}")

    print("=" * 50)

    if failures:
        print(f"\n{YELLOW}Some checks failed. The app may still work, but fix issues first.{RESET}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(run_all())

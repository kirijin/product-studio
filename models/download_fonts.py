"""Download Open Sans fonts from Google Fonts for label rendering."""

import os
import zipfile
import io
from pathlib import Path

FONTS_DIR = Path(__file__).resolve().parent / "fonts"

# Open Sans — variable font (replaces all static weights)
OPEN_SANS_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/"
)
OPEN_SANS_FILES = [
    "OpenSans[wdth,wght].ttf",
]

# Playfair Display — variable font
PLAYFAIR_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/"
)
PLAYFAIR_FILES = [
    "PlayfairDisplay[wght].ttf",
]


def _download_file(url, filename, dest_dir):
    """Download a single font file."""
    import urllib.request
    dest = dest_dir / filename
    if dest.exists():
        return True
    try:
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url + filename, str(dest))
        return True
    except Exception as e:
        print(f"  Failed to download {filename}: {e}")
        return False


def download_fonts():
    """Download all required font files. Returns True if all OK."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    print("Downloading fonts...")
    for fname in OPEN_SANS_FILES:
        ok = _download_file(OPEN_SANS_URL, fname, FONTS_DIR)
        all_ok = all_ok and ok

    for fname in PLAYFAIR_FILES:
        ok = _download_file(PLAYFAIR_URL, fname, FONTS_DIR)
        all_ok = all_ok and ok

    if all_ok:
        print(f"All fonts downloaded to {FONTS_DIR}")
    else:
        print("Some fonts failed — will fall back to default font.")
    return all_ok


def check_fonts_exist():
    """Return True if all required font files exist."""
    for fname in OPEN_SANS_FILES + PLAYFAIR_FILES:
        if not (FONTS_DIR / fname).exists():
            return False
    return True


if __name__ == "__main__":
    download_fonts()

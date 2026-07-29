"""Smart text placement using product mask to find safe zones."""

from PIL import Image
import numpy as np


def compute_text_size(text, font):
    """Return (width, height) of rendered text in pixels."""
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw_Dummy(img)
    # Use font.getbbox or getmask
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


class ImageDraw_Dummy:
    """Minimal stub — actual drawing happens in phase3_render.
    Here we only need measurement.
    """
    def __init__(self, img):
        self.img = img


def find_safe_placement(
    mask: Image.Image,
    text_width: int,
    text_height: int,
    position_preference: str = "auto",
    padding: int = 20,
    image_width: int = None,
    image_height: int = None,
):
    """Find the best (x, y) to place a text label without overlapping the product.

    Args:
        mask: Grayscale PIL image (white=product, black=background).
        text_width, text_height: Size of label including style padding.
        position_preference: 'auto', 'bottom-left', 'bottom-right',
                             'bottom-center', 'top-left', 'top-right', 'top-center'.
        padding: Minimum px from image edge.
        image_width, image_height: Fallback if mask size differs.

    Returns:
        (x, y, region_score) — top-left of the label placement.
        region_score 0..1 where 1 = completely free of product.
    """
    if image_width is None:
        image_width = mask.width
    if image_height is None:
        image_height = mask.height

    # Convert mask to numpy
    mask_np = np.array(mask.convert("L"))
    # Normalize to 0..1 (0 = background, 1 = product)
    mask_np = mask_np.astype(np.float32) / 255.0

    # Grid scoring — divide image into cells
    cell_size = 32
    cells_x = max(1, image_width // cell_size)
    cells_y = max(1, image_height // cell_size)

    # Resize mask to grid
    mask_grid = np.array(
        Image.fromarray(mask_np).resize((cells_x, cells_y), Image.NEAREST)
    )

    # Label size in cells
    label_cells_w = max(1, text_width // cell_size + 1)
    label_cells_h = max(1, text_height // cell_size + 1)

    # Define candidate regions based on position preference
    candidates = _get_position_candidates(
        position_preference, cells_x, cells_y, label_cells_w, label_cells_h
    )

    best_score = -1
    best_region = None

    for region in candidates:
        rx, ry = region
        # Clamp to grid boundaries
        if rx < 0 or ry < 0:
            continue
        if rx + label_cells_w > cells_x or ry + label_cells_h > cells_y:
            continue

        score = 1.0 - np.mean(mask_grid[ry:ry + label_cells_h, rx:rx + label_cells_w])
        if score > best_score:
            best_score = score
            best_region = (rx, ry)

    if best_region is None or best_score < 0.1:
        # No safe zone found — fallback to center-bottom with pill
        x = (image_width - text_width) // 2
        y = image_height - text_height - padding
        return (x, y, best_score)

    # Convert cell coords back to pixels
    rx, ry = best_region
    x = rx * cell_size
    y = ry * cell_size

    # Center the cell block around the actual text position
    x = x + (cell_size * label_cells_w - text_width) // 2
    y = y + (cell_size * label_cells_h - text_height) // 2

    # Clamp to image bounds
    x = max(padding, min(x, image_width - text_width - padding))
    y = max(padding, min(y, image_height - text_height - padding))

    return (int(x), int(y), float(best_score))


def _get_position_candidates(preference, cells_x, cells_y, label_w, label_h):
    """Return list of (cell_x, cell_y) candidates for the given preference."""
    margin = 1  # cell margin from edge

    regions = {
        "bottom-left": [(margin, cells_y - label_h - margin)],
        "bottom-right": [(cells_x - label_w - margin, cells_y - label_h - margin)],
        "bottom-center": [
            ((cells_x - label_w) // 2, cells_y - label_h - margin),
        ],
        "top-left": [(margin, margin)],
        "top-right": [(cells_x - label_w - margin, margin)],
        "top-center": [((cells_x - label_w) // 2, margin)],
    }

    if preference in regions:
        return regions[preference]

    # "auto" — try all positions, ranked by typical visual preference
    auto_candidates = [
        (cells_x - label_w - margin, cells_y - label_h - margin),  # bottom-right
        (margin, cells_y - label_h - margin),                      # bottom-left
        ((cells_x - label_w) // 2, cells_y - label_h - margin),    # bottom-center
        (cells_x - label_w - margin, margin),                      # top-right
        (margin, margin),                                          # top-left
        ((cells_x - label_w) // 2, margin),                        # top-center
    ]
    return auto_candidates

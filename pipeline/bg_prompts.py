"""Background generation prompt presets."""

BACKGROUND_PRESETS = {
    "studio": {
        "name": "Studio White",
        "prompt": (
            "clean white studio background, soft diffused lighting, "
            "professional product photography, high key, shallow depth of field, "
            "smooth gradient backdrop, 8K quality"
        ),
        "negative": (
            "people, text, watermark, logo, furniture, table, floor, "
            "cluttered, busy, shadows, grain, noise, low quality"
        ),
        "controlnet": "canny",
    },
    "studio_dark": {
        "name": "Studio Dark",
        "prompt": (
            "dark studio background, dramatic lighting, rim light on subject, "
            "professional product photography, dark gradient backdrop, "
            "high contrast, 8K quality"
        ),
        "negative": (
            "people, text, watermark, logo, bright background, "
            "cluttered, busy, grain, noise, low quality"
        ),
        "controlnet": "canny",
    },
    "outdoor_natural": {
        "name": "Outdoor Natural",
        "prompt": (
            "natural outdoor setting, soft daylight, wooden table surface, "
            "bokeh forest background, warm natural tones, "
            "professional product photography, 8K quality"
        ),
        "negative": (
            "people, text, watermark, logo, graffiti, trash, "
            "urban, cluttered, busy, low quality"
        ),
        "controlnet": "depth",
    },
    "marble": {
        "name": "Marble Surface",
        "prompt": (
            "white marble surface, soft studio lighting, "
            "luxury product photography, elegant minimal composition, "
            "subtle reflections, 8K quality"
        ),
        "negative": (
            "people, text, watermark, logo, table edge, "
            "cluttered, busy, grain, noise, low quality"
        ),
        "controlnet": "canny",
    },
    "outdoor_urban": {
        "name": "Urban Concrete",
        "prompt": (
            "urban concrete surface, raw industrial aesthetic, "
            "overcast natural lighting, minimalist composition, "
            "professional product photography, 8K quality"
        ),
        "negative": (
            "people, text, watermark, logo, graffiti, trash, "
            "grass, nature, bright colors, low quality"
        ),
        "controlnet": "depth",
    },
    "warm_interior": {
        "name": "Warm Interior",
        "prompt": (
            "warm wooden table in a cozy room, soft window light, "
            "home interior background, lifestyle product photography, "
            "inviting atmosphere, 8K quality"
        ),
        "negative": (
            "people, text, watermark, logo, cluttered, "
            "messy, cold lighting, low quality"
        ),
        "controlnet": "depth",
    },
}

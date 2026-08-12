"""Rebuild Figure 1 HRV Studio GUI screenshots with equal panel sizes."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SCREENSHOT_DIR = HERE / "HRV_Studio_Screenshots"

OUTPUT_PNG = HERE / "figure1_hrv_studio_gui.png"
OUTPUT_PDF = HERE / "figure1_hrv_studio_gui.pdf"
OUTPUT_SVG = HERE / "figure1_hrv_studio_gui.svg"

DPI = 300
CROP_BOX = (2, 55, 1401, 910)
PANEL_WIDTH = 2520
PANEL_HEIGHT = 1540
CANVAS_WIDTH = 2700
PANEL_X = (CANVAS_WIDTH - PANEL_WIDTH) // 2
PANEL_A_Y = 130
INTER_PANEL_GAP = 170
PANEL_B_Y = PANEL_A_Y + PANEL_HEIGHT + INTER_PANEL_GAP
BOTTOM_MARGIN = 110
CANVAS_HEIGHT = PANEL_B_Y + PANEL_HEIGHT + BOTTOM_MARGIN

LABEL_X = PANEL_X
LABEL_A_Y = 50
LABEL_B_Y = PANEL_A_Y + PANEL_HEIGHT + 88
LABEL_SIZE = 58
BORDER_WIDTH = 3

PANELS = [
    ("A", "Main analysis interface", SCREENSHOT_DIR / "Main_analysis_interface.png", PANEL_A_Y),
    ("B", "Beat editing interface", SCREENSHOT_DIR / "Beat_editing_interface (2).png", PANEL_B_Y),
]


def load_label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_panel(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.crop(CROP_BOX).resize(
            (PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS
        )


def encode_png_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def draw_raster_figure(panel_images: dict[str, Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_label_font(LABEL_SIZE)

    for label, _title, _path, panel_y in PANELS:
        canvas.paste(panel_images[label], (PANEL_X, panel_y))
        label_y = LABEL_A_Y if label == "A" else LABEL_B_Y
        draw.text((LABEL_X, label_y), f"{label}. {_title}", fill="black", font=font)
        draw.rectangle(
            [
                PANEL_X,
                panel_y,
                PANEL_X + PANEL_WIDTH - 1,
                panel_y + PANEL_HEIGHT - 1,
            ],
            outline="black",
            width=BORDER_WIDTH,
        )

    return canvas


def write_svg(panel_images: dict[str, Image.Image]) -> None:
    label_a_baseline = LABEL_A_Y + LABEL_SIZE
    label_b_baseline = LABEL_B_Y + LABEL_SIZE
    panel_a_uri = encode_png_data_uri(panel_images["A"])
    panel_b_uri = encode_png_data_uri(panel_images["B"])
    height_in = CANVAS_HEIGHT / DPI
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="9in" height="{height_in:.4f}in" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">
  <rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="white"/>
  <text x="{LABEL_X}" y="{label_a_baseline}" font-family="Arial, Helvetica, sans-serif" font-size="{LABEL_SIZE}" font-weight="700" fill="black">A. Main analysis interface</text>
  <image x="{PANEL_X}" y="{PANEL_A_Y}" width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" href="{panel_a_uri}"/>
  <rect x="{PANEL_X}" y="{PANEL_A_Y}" width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" fill="none" stroke="black" stroke-width="{BORDER_WIDTH}"/>
  <text x="{LABEL_X}" y="{label_b_baseline}" font-family="Arial, Helvetica, sans-serif" font-size="{LABEL_SIZE}" font-weight="700" fill="black">B. Beat editing interface</text>
  <image x="{PANEL_X}" y="{PANEL_B_Y}" width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" href="{panel_b_uri}"/>
  <rect x="{PANEL_X}" y="{PANEL_B_Y}" width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" fill="none" stroke="black" stroke-width="{BORDER_WIDTH}"/>
</svg>
"""
    OUTPUT_SVG.write_text(svg, encoding="utf-8")


def main() -> None:
    panel_images = {label: make_panel(path) for label, _title, path, _panel_y in PANELS}
    canvas = draw_raster_figure(panel_images)
    canvas.save(OUTPUT_PNG, dpi=(DPI, DPI))
    canvas.save(OUTPUT_PDF, "PDF", resolution=float(DPI))
    write_svg(panel_images)

    panel_area = PANEL_WIDTH * PANEL_HEIGHT
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_SVG}")
    print(f"Canvas: {CANVAS_WIDTH} x {CANVAS_HEIGHT} px")
    print(f"Panel A: {PANEL_WIDTH} x {PANEL_HEIGHT} px, area={panel_area}")
    print(f"Panel B: {PANEL_WIDTH} x {PANEL_HEIGHT} px, area={panel_area}")


if __name__ == "__main__":
    main()

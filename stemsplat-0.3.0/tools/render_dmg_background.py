#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


WIDTH = 560
HEIGHT = 360


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def make_background() -> Image.Image:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (8, 18, 23, 255))
    pixels = image.load()

    top = (5, 15, 20)
    bottom = (21, 56, 69)

    for y in range(HEIGHT):
        t = y / max(HEIGHT - 1, 1)
        row = tuple(int(lerp(top[i], bottom[i], t)) for i in range(3))
        for x in range(WIDTH):
            pixels[x, y] = (*row, 255)

    shafts = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    shafts_draw = ImageDraw.Draw(shafts)
    shaft_specs = [
        (70, 80, (0, 180, 210, 20)),
        (175, 95, (0, 150, 195, 18)),
        (305, 110, (0, 165, 205, 22)),
        (445, 100, (0, 190, 220, 18)),
    ]
    for center_x, spread, color in shaft_specs:
        shafts_draw.rectangle(
            (center_x - spread // 2, -40, center_x + spread // 2, HEIGHT + 40),
            fill=color,
        )
    shafts = shafts.filter(ImageFilter.GaussianBlur(34))
    image = Image.alpha_composite(image, shafts)

    glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse((110, 94, 450, 314), fill=(0, 145, 190, 34))
    glow = glow.filter(ImageFilter.GaussianBlur(58))
    image = Image.alpha_composite(image, glow)

    arrow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    arrow_draw = ImageDraw.Draw(arrow)
    arrow_color = (220, 245, 250, 235)
    arrow_shadow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(arrow_shadow)

    line_y = 188
    start_x = 225
    end_x = 336
    stroke = 12
    head = 26

    shadow_draw.line((start_x, line_y, end_x, line_y), fill=(0, 0, 0, 75), width=stroke)
    shadow_draw.line(
        (end_x - head, line_y - head + 2, end_x, line_y, end_x - head, line_y + head - 2),
        fill=(0, 0, 0, 75),
        width=stroke,
        joint="curve",
    )
    arrow_shadow = arrow_shadow.filter(ImageFilter.GaussianBlur(10))
    image = Image.alpha_composite(image, arrow_shadow)

    arrow_draw.line((start_x, line_y, end_x, line_y), fill=arrow_color, width=stroke)
    arrow_draw.line(
        (end_x - head, line_y - head, end_x, line_y, end_x - head, line_y + head),
        fill=arrow_color,
        width=stroke,
        joint="curve",
    )
    arrow = arrow.filter(ImageFilter.GaussianBlur(0.2))
    image = Image.alpha_composite(image, arrow)

    text_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)
    caption = "drag to applications"
    bbox = text_draw.textbbox((0, 0), caption)
    text_x = (WIDTH - (bbox[2] - bbox[0])) / 2
    text_y = 290
    text_draw.text((text_x, text_y), caption, fill=(232, 242, 245, 210))
    text_layer = text_layer.filter(ImageFilter.GaussianBlur(0.2))
    image = Image.alpha_composite(image, text_layer)

    return image


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: render_dmg_background.py OUTPUT_PATH", file=sys.stderr)
        return 1
    output_path = Path(sys.argv[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_background().save(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

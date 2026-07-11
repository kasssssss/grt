#!/usr/bin/env python3
"""Export latest or selected TensorBoard image summaries into PNG files."""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

from PIL import Image, ImageDraw
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def comparison_row_count(img: Image.Image) -> int:
    """Infer row count for GRT comparison_grid output."""
    if img.height % 4 == 0:
        return 4
    if img.height % 2 == 0:
        return 2
    return 0


def annotate_comparison_rows(img: Image.Image) -> Image.Image:
    rows = comparison_row_count(img)
    if rows == 0:
        return img

    margin = 54
    out = Image.new("RGB", (img.width + margin, img.height), "white")
    out.paste(img, (margin, 0))
    draw = ImageDraw.Draw(out)
    row_h = img.height // rows
    labels = ["GT" if i % 2 == 0 else "Pred" for i in range(rows)]
    for i, label in enumerate(labels):
        y0 = i * row_h
        y1 = (i + 1) * row_h
        color = (0, 0, 0) if label == "GT" else (180, 0, 0)
        draw.text((8, y0 + max(4, row_h // 2 - 6)), label, fill=color)
        draw.line((margin - 4, y0, margin - 4, y1), fill=(220, 220, 220))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--event-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-tags", type=int, default=64)
    p.add_argument(
        "--steps", nargs="*", type=int,
        help="Export these exact summary steps instead of only the latest.")
    p.add_argument(
        "--tags", nargs="*",
        help="Restrict export to these exact image tags.")
    p.add_argument(
        "--contact-name", default="latest_tensorboard_images_contact_sheet.png")
    args = p.parse_args()

    event_dir = Path(args.event_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ea = EventAccumulator(str(event_dir), size_guidance={"images": 0, "scalars": 1})
    ea.Reload()
    image_tags = ea.Tags().get("images", [])
    scalar_tags = ea.Tags().get("scalars", [])
    print("IMAGE_TAGS", image_tags)
    print("SCALAR_TAGS", scalar_tags[:80])

    exported = []
    selected_tags = image_tags if args.tags is None else [
        tag for tag in image_tags if tag in args.tags]
    for tag in selected_tags[: args.max_tags]:
        images = ea.Images(tag)
        if not images:
            continue
        if args.steps is None:
            selected = [images[-1]]
        else:
            by_step = {event.step: event for event in images}
            missing = [step for step in args.steps if step not in by_step]
            if missing:
                print("MISSING_STEPS", tag, missing)
            selected = [by_step[step] for step in args.steps if step in by_step]
        for ev in selected:
            img = Image.open(io.BytesIO(ev.encoded_image_string)).convert("RGB")
            img = annotate_comparison_rows(img)
            out = out_dir / f"{slug(tag)}_step{ev.step}.png"
            img.save(out)
            exported.append((tag, ev.step, out.name, img.size))
            print("EXPORTED", tag, ev.step, out, img.size)

    if exported:
        thumbs = []
        label_h = 24
        cell_w = max(img_size[0] for _, _, _, img_size in exported)
        cell_h = max(img_size[1] for _, _, _, img_size in exported) + label_h
        for tag, step, name, _ in exported:
            img = Image.open(out_dir / name).convert("RGB")
            cell = Image.new("RGB", (cell_w, cell_h), "white")
            cell.paste(img, ((cell_w - img.width) // 2, label_h))
            draw = ImageDraw.Draw(cell)
            draw.text((4, 4), f"{tag} @ {step}", fill=(0, 0, 0))
            thumbs.append(cell)
        cols = 2 if args.steps and len(thumbs) > 1 else min(3, len(thumbs))
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
        for i, img in enumerate(thumbs):
            sheet.paste(img, ((i % cols) * cell_w, (i // cols) * cell_h))
        sheet_path = out_dir / args.contact_name
        sheet.save(sheet_path)
        print("CONTACT_SHEET", sheet_path, sheet.size)
    else:
        print("NO_IMAGES_EXPORTED")


if __name__ == "__main__":
    main()

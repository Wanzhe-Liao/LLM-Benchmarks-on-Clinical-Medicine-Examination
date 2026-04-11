from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import fitz  # type: ignore
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import Color, black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


@dataclass(frozen=True)
class PanelPlacement:
    label: str
    x: float
    y: float
    width: float
    height: float


def _first_page(path: Path):
    reader = PdfReader(str(path))
    if not reader.pages:
        raise ValueError(f"PDF has no pages: {path}")
    return reader.pages[0]


def _page_size(page) -> Tuple[float, float]:
    box = page.mediabox
    return float(box.width), float(box.height)


def _make_label_overlay_pdf(
    page_width: float,
    page_height: float,
    panels: Iterable[PanelPlacement],
    *,
    font_name: str,
    font_size: float,
    inset: float,
    box_pad: float,
    box_alpha: float,
) -> bytes:
    # Use a single font family across the entire figure set (Matplotlib + ReportLab overlays).
    # ReportLab only ships core PDF fonts by default; register Arial from the OS when available.
    if font_name == "Arial" and "Arial" not in pdfmetrics.getRegisteredFontNames():
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        candidates = [
            windir / "Fonts" / "arial.ttf",
            windir / "Fonts" / "ARIAL.TTF",
        ]
        for fp in candidates:
            if not fp.exists():
                continue
            try:
                pdfmetrics.registerFont(TTFont("Arial", str(fp)))
                break
            except Exception:
                continue
        if "Arial" not in pdfmetrics.getRegisteredFontNames():
            font_name = "Helvetica"

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_width, page_height))
    c.setFont(font_name, font_size)

    for panel in panels:
        x = panel.x + inset
        y = panel.y + panel.height - inset - font_size

        text_w = float(pdfmetrics.stringWidth(panel.label, font_name, font_size))
        box_w = text_w + 2.0 * box_pad
        box_h = font_size + 2.0 * box_pad

        c.saveState()
        c.setFillColor(Color(1, 1, 1, alpha=box_alpha))
        c.setStrokeColor(Color(1, 1, 1, alpha=0))
        c.rect(x - box_pad, y - box_pad, box_w, box_h, fill=1, stroke=0)
        c.restoreState()

        c.setFillColor(black)
        c.drawString(x, y, panel.label)

    c.showPage()
    c.save()
    return buf.getvalue()


def compose_two_panel_figure(
    *,
    a_pdf: Path,
    b_pdf: Path,
    out_pdf: Path,
    out_png: Path,
    dpi: int = 300,
    margin: float = 24.0,
    gap: float = 18.0,
    min_landscape_ratio: float = 0.0,
    label_a: str = "A",
    label_b: str = "B",
    label_font: str = "Arial",
    label_font_size: float = 18.0,
    label_inset: float = 12.0,
    label_box_pad: float = 4.0,
    label_box_alpha: float = 0.85,
) -> None:
    page_a = _first_page(a_pdf)
    page_b = _first_page(b_pdf)

    wa, ha = _page_size(page_a)
    wb, hb = _page_size(page_b)

    # Normalize widths so panels align cleanly in a vertical stack.
    target_w = max(wa, wb)
    scale_a = (target_w / wa) if wa else 1.0
    scale_b = (target_w / wb) if wb else 1.0

    scaled_ha = ha * scale_a
    scaled_hb = hb * scale_b

    content_w = target_w
    content_h = scaled_ha + gap + scaled_hb

    page_h = content_h + 2.0 * margin
    page_w = content_w + 2.0 * margin
    if float(min_landscape_ratio) > 0:
        page_w = max(page_w, page_h * float(min_landscape_ratio))

    content_left = (page_w - content_w) * 0.5

    b_x = content_left + (content_w - (wb * scale_b)) * 0.5
    b_y = margin
    a_x = content_left + (content_w - (wa * scale_a)) * 0.5
    a_y = margin + scaled_hb + gap

    writer = PdfWriter()
    out_page = writer.add_blank_page(width=page_w, height=page_h)

    out_page.merge_transformed_page(
        page_a,
        Transformation().scale(scale_a).translate(a_x, a_y),
    )
    out_page.merge_transformed_page(
        page_b,
        Transformation().scale(scale_b).translate(b_x, b_y),
    )

    overlay_bytes = _make_label_overlay_pdf(
        page_w,
        page_h,
        [
            PanelPlacement(label=label_a, x=a_x, y=a_y, width=wa * scale_a, height=scaled_ha),
            PanelPlacement(label=label_b, x=b_x, y=b_y, width=wb * scale_b, height=scaled_hb),
        ],
        font_name=label_font,
        font_size=label_font_size,
        inset=label_inset,
        box_pad=label_box_pad,
        box_alpha=label_box_alpha,
    )
    overlay_page = PdfReader(io.BytesIO(overlay_bytes)).pages[0]
    out_page.merge_page(overlay_page)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as f:
        writer.write(f)

    # Render PNG preview from the composed PDF for consistent appearance.
    doc = fitz.open(str(out_pdf))
    try:
        page = doc.load_page(0)
        zoom = float(dpi) / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_png))
    finally:
        doc.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compose a 2-panel (A/B) figure into one PDF+PNG.")
    ap.add_argument("--a-pdf", required=True, type=Path, help="Panel A PDF (single-page).")
    ap.add_argument("--b-pdf", required=True, type=Path, help="Panel B PDF (single-page).")
    ap.add_argument("--out-pdf", required=True, type=Path, help="Output PDF path.")
    ap.add_argument("--out-png", type=Path, help="Output PNG path (default: same name as out-pdf).")
    ap.add_argument("--dpi", type=int, default=300, help="PNG render DPI.")
    ap.add_argument("--margin", type=float, default=24.0, help="Outer margin (pt).")
    ap.add_argument("--gap", type=float, default=18.0, help="Gap between panels (pt).")
    ap.add_argument(
        "--min-landscape-ratio",
        type=float,
        default=1.03,
        help="Enforce width >= height * ratio.",
    )
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--label-font-size", type=float, default=18.0)
    ap.add_argument("--label-inset", type=float, default=12.0)

    args = ap.parse_args()

    out_png = args.out_png
    if out_png is None:
        out_png = args.out_pdf.with_suffix(".png")

    compose_two_panel_figure(
        a_pdf=args.a_pdf,
        b_pdf=args.b_pdf,
        out_pdf=args.out_pdf,
        out_png=out_png,
        dpi=int(args.dpi),
        margin=float(args.margin),
        gap=float(args.gap),
        min_landscape_ratio=float(args.min_landscape_ratio),
        label_a=str(args.label_a),
        label_b=str(args.label_b),
        label_font_size=float(args.label_font_size),
        label_inset=float(args.label_inset),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

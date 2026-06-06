"""
Axiom — PDF Report Generator

Builds a polished, multi-page PDF from pipeline state, results, and embedded
visualizations using ReportLab's Platypus framework. No system dependencies.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Brand palette ──────────────────────────────────────────────────────────

ACCENT = HexColor("#00b39a")          # darker teal — better print contrast than #00e5c8
ACCENT_LIGHT = HexColor("#e6fbf7")
INK = HexColor("#0f172a")
TEXT = HexColor("#1f2937")
MUTED = HexColor("#6b7280")
GHOST = HexColor("#9ca3af")
HAIRLINE = HexColor("#e5e7eb")
SUCCESS = HexColor("#10b981")
WARNING = HexColor("#f59e0b")
DESTRUCTIVE = HexColor("#dc2626")
SURFACE = HexColor("#f9fafb")

PAGE_W, PAGE_H = LETTER
MARGIN_X = 0.7 * inch
MARGIN_TOP = 0.9 * inch
MARGIN_BOTTOM = 0.7 * inch


# ── Stylesheet ─────────────────────────────────────────────────────────────

def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    return {
        "h_brand": ParagraphStyle(
            "h_brand", parent=base, fontName="Helvetica-Bold", fontSize=10,
            textColor=ACCENT, leading=12, spaceAfter=4, letterSpacing=2,
        ),
        "h_title": ParagraphStyle(
            "h_title", parent=base, fontName="Helvetica-Bold", fontSize=28,
            textColor=INK, leading=32, spaceAfter=6,
        ),
        "h_section": ParagraphStyle(
            "h_section", parent=base, fontName="Helvetica-Bold", fontSize=14,
            textColor=INK, leading=18, spaceBefore=20, spaceAfter=8,
        ),
        "h_subsection": ParagraphStyle(
            "h_subsection", parent=base, fontName="Helvetica-Bold", fontSize=10,
            textColor=ACCENT, leading=12, spaceBefore=12, spaceAfter=4,
            letterSpacing=1.5,
        ),
        "lede": ParagraphStyle(
            "lede", parent=base, fontName="Helvetica", fontSize=11,
            textColor=MUTED, leading=16, spaceAfter=10,
        ),
        "body": ParagraphStyle(
            "body", parent=base, fontName="Helvetica", fontSize=10,
            textColor=TEXT, leading=14, spaceAfter=6,
        ),
        "kpi_value": ParagraphStyle(
            "kpi_value", parent=base, fontName="Helvetica-Bold", fontSize=20,
            textColor=INK, leading=22, alignment=TA_LEFT,
        ),
        "kpi_label": ParagraphStyle(
            "kpi_label", parent=base, fontName="Helvetica-Bold", fontSize=7,
            textColor=MUTED, leading=10, letterSpacing=1.2, alignment=TA_LEFT,
        ),
        "mono": ParagraphStyle(
            "mono", parent=base, fontName="Courier", fontSize=9,
            textColor=ACCENT, leading=12,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base, fontName="Helvetica-Oblique", fontSize=8,
            textColor=GHOST, leading=10, alignment=TA_CENTER, spaceBefore=4,
        ),
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def _kpi_card(label: str, value: str, styles: dict) -> Table:
    """A single bordered KPI tile."""
    inner = Table(
        [[Paragraph(value, styles["kpi_value"])],
         [Paragraph(label.upper(), styles["kpi_label"])]],
        colWidths=[1.6 * inch], rowHeights=[0.36 * inch, 0.22 * inch],
    )
    inner.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("LINEABOVE", (0, 0), (-1, 0), 2, ACCENT),
        ("BOX", (0, 0), (-1, -1), 0.5, HAIRLINE),
    ]))
    return inner


def _kpi_row(items: list[tuple[str, str]], styles: dict) -> Table:
    """A horizontal strip of KPI cards (up to 4)."""
    cells = [_kpi_card(label, value, styles) for label, value in items]
    # Pad to 4 columns for consistent spacing
    while len(cells) < 4:
        cells.append("")
    row = Table([cells], colWidths=[1.75 * inch] * 4)
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return row


def _hero_card(result: dict, styles: dict) -> Table:
    """Big hero card with the best model + headline metric."""
    best_model = result.get("best_model") or "—"
    metric_name = result.get("best_metric_name") or "score"
    metric_value = result.get("best_metric_value")
    metric_str = _fmt(metric_value) if metric_value is not None else "—"
    problem_type = (result.get("problem_type") or "—").replace("_", " ").title()
    target = result.get("target_column") or "—"

    left = [
        [Paragraph("CHAMPION MODEL", styles["kpi_label"])],
        [Paragraph(best_model, ParagraphStyle(
            "hero_model", parent=styles["body"], fontName="Helvetica-Bold",
            fontSize=22, textColor=INK, leading=26, spaceAfter=8,
        ))],
        [Paragraph(
            f'<font color="#00b39a"><b>{metric_name}</b></font> '
            f'<font color="#0f172a"><b>{metric_str}</b></font>',
            ParagraphStyle("hero_metric", parent=styles["body"], fontSize=13,
                           leading=16, textColor=TEXT),
        )],
        [Paragraph(
            f"<b>Problem:</b> {problem_type}&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"<b>Target:</b> {target}",
            ParagraphStyle("hero_meta", parent=styles["body"], fontSize=9,
                           leading=12, textColor=MUTED, spaceBefore=6),
        )],
    ]
    hero = Table(left, colWidths=[6.2 * inch])
    hero.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_LIGHT),
        ("LINEABOVE", (0, 0), (-1, 0), 3, ACCENT),
        ("BOX", (0, 0), (-1, -1), 0.5, HAIRLINE),
    ]))
    return hero


def _section_header(text: str, styles: dict) -> Paragraph:
    return Paragraph(text, styles["h_section"])


def _eyebrow(text: str, styles: dict) -> Paragraph:
    return Paragraph(text.upper(), styles["h_subsection"])


def _kv_table(rows: list[tuple[str, Any]]) -> Table:
    data = [[k, _fmt(v) if not isinstance(v, str) else v] for k, v in rows]
    if not data:
        data = [["—", "—"]]
    t = Table(data, colWidths=[2.2 * inch, 3.7 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, HAIRLINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _model_table(models: list[dict], styles: dict) -> Table:
    rows = [["#", "Model", "Status", "Metric", "Train Time"]]
    sorted_models = sorted(
        models,
        key=lambda m: (
            0 if m.get("is_best") else 1,
            -(list((m.get("metrics") or {}).values())[0] if m.get("metrics") else 0),
        ),
    )
    for i, m in enumerate(sorted_models, 1):
        name = m.get("name", "—")
        if m.get("is_best"):
            name = f"<b>★ {name}</b>"
        metrics = m.get("metrics") or {}
        metric_str = " · ".join(f"{k}: {v:.4f}" for k, v in list(metrics.items())[:2]) if metrics else "—"
        status = m.get("status") or "—"
        rows.append([
            str(i),
            Paragraph(name, styles["body"]),
            status,
            Paragraph(metric_str, styles["body"]),
            f"{m.get('time_s', 0):.2f}s" if m.get("time_s") is not None else "—",
        ])

    t = Table(rows, colWidths=[0.4 * inch, 1.9 * inch, 0.9 * inch, 3.0 * inch, 0.8 * inch])
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("BACKGROUND", (0, 0), (-1, 0), SURFACE),
        ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, HAIRLINE),
    ]
    # Highlight the best model row (first data row after sort)
    if len(rows) > 1:
        style.append(("BACKGROUND", (0, 1), (-1, 1), ACCENT_LIGHT))
    t.setStyle(TableStyle(style))
    return t


def _inline_md(text: str) -> str:
    """Escape a markdown cell/line and apply inline **bold** / `code` markup."""
    import re
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"`([^`]+)`", r'<font face="Courier" color="#00b39a">\1</font>', safe)
    return safe


def _md_table(table_lines: list[str], styles: dict) -> Optional[Table]:
    """Render a GFM markdown table (list of ``| a | b |`` lines) as a styled Table.

    Previously the narrative appendix dumped these lines as raw text, so every
    table showed as literal ``| ... |`` pipes in the PDF. This parses them into
    real, themed ReportLab tables matching the rest of the report.
    """
    def parse_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    def is_sep(cells: list[str]) -> bool:
        return bool(cells) and all(set(c) <= set(":- ") and "-" in c for c in cells)

    rows_raw = [parse_row(l) for l in table_lines if l.strip().startswith("|")]
    rows_raw = [r for r in rows_raw if not is_sep(r)]
    if not rows_raw:
        return None

    ncols = max(len(r) for r in rows_raw)
    head_style = ParagraphStyle(
        "td_head", parent=styles["body"], fontName="Helvetica-Bold",
        fontSize=8, textColor=MUTED, leading=11,
    )
    cell_style = ParagraphStyle(
        "td_cell", parent=styles["body"], fontName="Helvetica",
        fontSize=9, textColor=TEXT, leading=12,
    )

    data: list[list] = []
    for ri, r in enumerate(rows_raw):
        r = r + [""] * (ncols - len(r))
        style = head_style if ri == 0 else cell_style
        data.append([Paragraph(_inline_md(c), style) for c in r])

    avail = PAGE_W - 2 * MARGIN_X
    col_w = avail / ncols
    t = Table(data, colWidths=[col_w] * ncols, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SURFACE),
        ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SURFACE]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, HAIRLINE),
        ("BOX", (0, 0), (-1, -1), 0.5, HAIRLINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _shap_table(shap_items: list[tuple[str, float]], styles: dict) -> Table:
    rows: list[list[Any]] = [["Rank", "Feature", "Mean |SHAP|"]]
    for i, (feature, value) in enumerate(shap_items[:15], 1):
        rows.append([f"#{i}", feature, f"{value:.6f}"])
    t = Table(rows, colWidths=[0.6 * inch, 4.0 * inch, 2.4 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("BACKGROUND", (0, 0), (-1, 0), SURFACE),
        ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
        ("FONTNAME", (0, 1), (0, -1), "Courier"),
        ("FONTNAME", (2, 1), (2, -1), "Courier-Bold"),
        ("TEXTCOLOR", (2, 1), (2, -1), ACCENT),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (2, 1), (2, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, HAIRLINE),
    ]))
    return t


def _errors_block(errors: list[dict], styles: dict) -> list:
    if not errors:
        success_box = Table(
            [[Paragraph("Clean run — no issues detected", ParagraphStyle(
                "ok", parent=styles["body"], fontName="Helvetica-Bold", fontSize=11,
                textColor=SUCCESS, leading=14, alignment=TA_CENTER,
            ))]],
            colWidths=[6.2 * inch], rowHeights=[0.55 * inch],
        )
        success_box.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#ecfdf5")),
            ("LINEABOVE", (0, 0), (-1, 0), 2, SUCCESS),
            ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#a7f3d0")),
        ]))
        return [success_box]

    flow = []
    for err in errors:
        severity = (err.get("severity") or "info").lower()
        color = DESTRUCTIVE if severity == "critical" else WARNING if severity == "warning" else ACCENT
        bg = HexColor("#fef2f2") if severity == "critical" else HexColor("#fffbeb") if severity == "warning" else ACCENT_LIGHT
        rows = [
            [Paragraph(
                f"<b>{(err.get('type') or 'issue').replace('_', ' ').title()}</b> "
                f'<font color="#6b7280"> · {severity.upper()}</font>',
                ParagraphStyle("err_h", parent=styles["body"], fontSize=10,
                               textColor=color, leading=14),
            )],
            [Paragraph(err.get("cause") or "—", styles["body"])],
        ]
        if err.get("fix"):
            rows.append([Paragraph(
                f"→ {err.get('fix')}",
                ParagraphStyle("err_fix", parent=styles["body"], fontSize=9,
                               textColor=ACCENT, leading=12),
            )])
        t = Table(rows, colWidths=[6.2 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LINEBEFORE", (0, 0), (0, -1), 3, color),
        ]))
        flow.append(t)
        flow.append(Spacer(1, 6))
    return flow


def _embed_viz(viz: dict, styles: dict, max_w: float = 6.2 * inch) -> Optional[list]:
    """Decode a base64 PNG viz into an Image flowable + caption."""
    b64 = viz.get("base64_png")
    if not b64:
        return None
    try:
        img_data = base64.b64decode(b64)
        bio = io.BytesIO(img_data)
        img = Image(bio)
        # Scale to fit max_w while preserving aspect ratio
        iw, ih = img.imageWidth, img.imageHeight
        scale = max_w / iw if iw else 1
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        return [
            img,
            Paragraph(
                f"<b>{viz.get('name', '')}</b> — {viz.get('description', '')}",
                styles["caption"],
            ),
            Spacer(1, 10),
        ]
    except Exception:
        return None


# ── Page chrome (header/footer) ────────────────────────────────────────────

def _make_page_chrome(run_id: str, mode: str):
    def draw(canvas: Canvas, doc: BaseDocTemplate) -> None:
        canvas.saveState()
        # Top accent bar
        canvas.setFillColor(ACCENT)
        canvas.rect(0, PAGE_H - 4, PAGE_W, 4, fill=1, stroke=0)
        # Header row
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(INK)
        canvas.drawString(MARGIN_X, PAGE_H - 32, "AXIOM")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN_X + 48, PAGE_H - 32, "Autonomous Data Scientist")
        canvas.setFont("Courier", 8)
        canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 32, run_id)
        # Hairline below header
        canvas.setStrokeColor(HAIRLINE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN_X, PAGE_H - 40, PAGE_W - MARGIN_X, PAGE_H - 40)
        # Footer
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GHOST)
        canvas.drawString(MARGIN_X, 24,
                          f"Mode: {mode.capitalize()}  ·  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas.drawRightString(PAGE_W - MARGIN_X, 24, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()
    return draw


# ── Main entry point ───────────────────────────────────────────────────────

def build_pdf(
    run_info: dict,
    result: dict,
    visualizations: Optional[list[dict]] = None,
    shap_data: Optional[dict[str, float]] = None,
    markdown_report: Optional[str] = None,
) -> bytes:
    """Render the full pipeline run to a styled PDF and return bytes."""
    visualizations = visualizations or []
    styles = _styles()

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=LETTER,
        leftMargin=MARGIN_X, rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
        title="Axiom Pipeline Report",
        author="Axiom Autonomous Data Scientist",
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height, id="normal", showBoundary=0,
    )
    run_id = run_info.get("run_id") or result.get("run_id") or "—"
    mode = run_info.get("mode") or "free"
    doc.addPageTemplates([PageTemplate(
        id="default", frames=[frame], onPage=_make_page_chrome(run_id, mode),
    )])

    story: list = []

    # ── Cover header ───────────────────────────────────────────────────────
    story.append(Paragraph("PIPELINE REPORT", styles["h_brand"]))
    story.append(Paragraph("Execution Summary", styles["h_title"]))
    story.append(Paragraph(
        "Autonomous end-to-end machine-learning workflow — every stage profiled, "
        "every model evaluated, every artifact catalogued.",
        styles["lede"],
    ))
    story.append(Spacer(1, 8))

    # ── Hero card ──────────────────────────────────────────────────────────
    story.append(_hero_card(result, styles))
    story.append(Spacer(1, 16))

    # ── KPI strip ──────────────────────────────────────────────────────────
    ds = result.get("dataset") or {}
    models = result.get("models") or []
    kpis = [
        ("Models Trained", str(len([m for m in models if m.get("status") == "trained"]))),
        ("Dataset Rows", _fmt_int(ds.get("rows"))),
        ("Retries", _fmt_int(result.get("retry_count"))),
        ("Duration", _fmt_duration(run_info.get("duration_seconds"))),
    ]
    story.append(_kpi_row(kpis, styles))
    story.append(Spacer(1, 8))

    # ── Dataset & Preprocessing ────────────────────────────────────────────
    story.append(_section_header("Dataset & Preprocessing", styles))
    pp = result.get("preprocessing") or {}
    fe = result.get("features") or {}
    left_col = _kv_table([
        ("Rows", ds.get("rows")),
        ("Columns", ds.get("columns")),
        ("Quality Score", ds.get("quality_score")),
        ("Target", result.get("target_column") or "None"),
        ("Problem Type", result.get("problem_type")),
    ])
    right_col = _kv_table([
        ("Rows Before", pp.get("rows_before")),
        ("Rows After", pp.get("rows_after")),
        ("Duplicates Removed", pp.get("duplicates_removed")),
        ("Quality Score", pp.get("quality_score")),
        ("Features Before → After",
         f"{fe.get('before', '—')} → {fe.get('after', '—')}"),
    ])
    pair = Table(
        [[left_col, right_col]],
        colWidths=[3.0 * inch, 3.2 * inch],
    )
    pair.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, -1), 20),
    ]))
    story.append(pair)

    selected = fe.get("selected") or []
    if selected:
        story.append(Spacer(1, 10))
        story.append(_eyebrow("Selected Features", styles))
        chips = ", ".join(selected[:24])
        if len(selected) > 24:
            chips += f", +{len(selected) - 24} more"
        story.append(Paragraph(
            chips,
            ParagraphStyle("chips", parent=styles["body"], fontName="Courier",
                           fontSize=8.5, textColor=ACCENT, leading=12),
        ))

    # ── Model leaderboard ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(_section_header("Model Leaderboard", styles))
    story.append(Paragraph(
        f"{len([m for m in models if m.get('status') == 'trained'])} models trained · "
        f"champion highlighted in teal.",
        styles["lede"],
    ))
    if models:
        story.append(_model_table(models, styles))
    else:
        story.append(Paragraph("No models trained in this run.", styles["body"]))

    # Model comparison viz, if available
    model_viz = next(
        (v for v in visualizations if "model" in (v.get("name") or "").lower()
         or v.get("type") == "model_comparison"),
        None,
    )
    if model_viz:
        story.append(Spacer(1, 14))
        story.append(_eyebrow("Model Performance Chart", styles))
        embedded = _embed_viz(model_viz, styles)
        if embedded:
            story.extend(embedded)

    # ── Explainability ─────────────────────────────────────────────────────
    if shap_data:
        story.append(PageBreak())
        story.append(_section_header("Explainability — SHAP", styles))
        story.append(Paragraph(
            "SHapley Additive exPlanations attribute each feature's contribution to model "
            "predictions. Larger mean |SHAP| values indicate higher influence on the output.",
            styles["lede"],
        ))
        shap_items = sorted(shap_data.items(), key=lambda kv: kv[1], reverse=True)
        story.append(_shap_table(shap_items, styles))

        feature_viz = next(
            (v for v in visualizations if "shap" in (v.get("name") or "").lower()
             or "feature_importance" in (v.get("type") or "").lower()),
            None,
        )
        if feature_viz:
            story.append(Spacer(1, 14))
            embedded = _embed_viz(feature_viz, styles)
            if embedded:
                story.extend(embedded)

    # ── Errors / Warnings ──────────────────────────────────────────────────
    errors = result.get("errors") or []
    story.append(PageBreak())
    story.append(_section_header("Quality Audit", styles))
    story.append(Paragraph(
        "Findings from the Error Detection agent — overfitting, leakage, imbalance, and "
        "data quality flags.",
        styles["lede"],
    ))
    story.extend(_errors_block(errors, styles))

    # ── Additional visualizations ──────────────────────────────────────────
    extra_viz = [
        v for v in visualizations
        if v is not model_viz
        and "shap" not in (v.get("name") or "").lower()
        and "feature_importance" not in (v.get("type") or "").lower()
    ]
    if extra_viz:
        story.append(PageBreak())
        story.append(_section_header("Visualizations", styles))
        for v in extra_viz[:6]:
            embedded = _embed_viz(v, styles)
            if embedded:
                story.append(KeepTogether(embedded))

    # ── Markdown report appendix ───────────────────────────────────────────
    if markdown_report:
        story.append(PageBreak())
        story.append(_section_header("Full Pipeline Narrative", styles))
        story.append(Paragraph(
            "Auto-generated agent narrative for the run.",
            styles["lede"],
        ))
        md_lines = markdown_report.split("\n")
        j = 0
        while j < len(md_lines):
            line = md_lines[j].rstrip()
            stripped = line.strip()

            # GFM table block — collect consecutive pipe lines and render as a table
            if stripped.startswith("|"):
                table_lines = []
                while j < len(md_lines) and md_lines[j].strip().startswith("|"):
                    table_lines.append(md_lines[j])
                    j += 1
                tbl = _md_table(table_lines, styles)
                if tbl is not None:
                    story.append(Spacer(1, 4))
                    story.append(KeepTogether(tbl))
                    story.append(Spacer(1, 8))
                continue

            if not line:
                story.append(Spacer(1, 4))
            elif stripped.startswith("---"):
                story.append(Spacer(1, 2))
            elif line.startswith("# "):
                story.append(Paragraph(line[2:].strip(), ParagraphStyle(
                    "md_h1", parent=styles["body"], fontName="Helvetica-Bold",
                    fontSize=14, textColor=INK, leading=18, spaceBefore=10, spaceAfter=4,
                )))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:].strip(), ParagraphStyle(
                    "md_h2", parent=styles["body"], fontName="Helvetica-Bold",
                    fontSize=11, textColor=ACCENT, leading=14, spaceBefore=8, spaceAfter=3,
                )))
            elif line.startswith("### "):
                story.append(Paragraph(line[4:].strip(), ParagraphStyle(
                    "md_h3", parent=styles["body"], fontName="Helvetica-Bold",
                    fontSize=10, textColor=INK, leading=12, spaceBefore=6, spaceAfter=2,
                )))
            elif line.startswith("- ") or line.startswith("* "):
                story.append(Paragraph(
                    f"• {_inline_md(line[2:].strip())}",
                    ParagraphStyle("md_li", parent=styles["body"], leftIndent=12,
                                   fontSize=9.5, leading=13),
                ))
            else:
                story.append(Paragraph(_inline_md(line), styles["body"]))
            j += 1

    doc.build(story)
    return buf.getvalue()

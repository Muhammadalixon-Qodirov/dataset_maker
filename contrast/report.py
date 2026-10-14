"""PDF output: the coaching report of one analysis, and the technical documentation.

    analysis_pdf(...)   rubric, profile, practice plan, word heatmap, every flaw + tip
    markdown_pdf(md)    contrast/METHOD.md -> PDF (python -m contrast.report)

The report is rendered from the same dicts the dashboard shows, so the two never disagree.
"""

from __future__ import annotations

import os
import re

from fpdf import FPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(ROOT, "assets", "fonts", "IBMPlexSans.ttf")
INK, DIM, ACC = (16, 34, 28), (74, 90, 82), (31, 122, 82)


def _pdf() -> FPDF:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_font("Plex", "", FONT)
    pdf.add_font("Plex", "B", FONT)  # variable font: same file; headings use size and color
    pdf.add_font("Plex", "I", FONT)
    pdf.set_margins(16, 16, 16)
    pdf.add_page()
    return pdf


def _h(pdf: FPDF, text: str, size: int = 14) -> None:
    pdf.ln(3)
    pdf.set_font("Plex", "", size)
    pdf.set_text_color(*ACC)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, size * 0.5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*INK)
    pdf.ln(1)


def _p(pdf: FPDF, text: str, size: float = 10, color=INK) -> None:
    pdf.set_font("Plex", "", size)
    pdf.set_text_color(*color)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, size * 0.48, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*INK)


def _heat_rgb(h: float) -> tuple[int, int, int]:
    # green (fine) -> amber -> red (flaw); dark enough to read on white
    if h < 0.35:
        return (31, 122, 82)
    if h < 0.7:
        return (183, 121, 31)
    return (194, 65, 12)


def analysis_pdf(path: str, *, reference: str, participant: str, rubric: dict, findings: list[dict],
                 plan: list[dict], heat: list[dict], regions: list[dict]) -> str:
    pdf = _pdf()
    pdf.set_font("Plex", "", 22)
    pdf.set_text_color(*INK)
    pdf.cell(0, 11, "Speech Delivery Report", new_x="LMARGIN", new_y="NEXT")
    _p(pdf, f"Reference: {reference}\nParticipant: {participant}", 9.5, DIM)

    _h(pdf, f"Overall {rubric['overall']:.1f} / 10")
    pdf.set_font("Plex", "", 10)
    w = (pdf.w - 32) / len(rubric["criteria"])
    for c in rubric["criteria"]:
        pdf.cell(w, 6, c, border="B", align="C")
    pdf.ln()
    for v in rubric["criteria"].values():
        pdf.set_text_color(*_heat_rgb(1 - v / 10))
        pdf.set_font("Plex", "", 16)
        pdf.cell(w, 10, f"{v:.1f}", align="C")
    pdf.set_text_color(*INK)
    pdf.ln(12)

    if plan:
        _h(pdf, "Practise these first")
        for k, it in enumerate(plan, 1):
            _p(pdf, f"{k}. {it['what']}  ({it['criterion']}, about {it['points']:.1f} pts)", 10.5)
            _p(pdf, f"    {it['do']}", 9.5, DIM)
            pdf.ln(1)

    if findings:
        _h(pdf, "Your overall style vs the reference")
        for f in findings:
            _p(pdf, f"• {f['text']}  {f['tip']}", 9.5)

    _h(pdf, "Transcript heatmap")
    _p(pdf, "green = like the reference · amber = noticeable difference · red = flagged flaw", 8.5, DIM)
    pdf.ln(1)
    pdf.set_font("Plex", "", 11)
    for wd in heat:
        if pdf.get_x() + pdf.get_string_width(wd["word"]) > pdf.w - pdf.r_margin:
            pdf.ln(6)  # never break inside a word
        pdf.set_text_color(*_heat_rgb(wd["heat"]))
        pdf.write(6, wd["word"] + " ")
    pdf.set_text_color(*INK)
    pdf.ln(9)

    _h(pdf, f"Every flaw ({len(regions)})")
    if not regions:
        _p(pdf, "No local flaws detected against the reference.")
    for r in regions:
        _p(pdf, f"{r['start']:.2f}–{r['end']:.2f} s · {r['label']} · {r['criterion']}", 10.5, ACC)
        _p(pdf, f"“{r['words']}”", 9.5, DIM)
        _p(pdf, f"Why: {r['explanation']}", 9.5)
        _p(pdf, f"Do: {r['tip']}", 9.5)
        pdf.ln(2)

    _p(pdf, "\nScores are deterministic: the same audio and transcript give the same report. "
            "Method: contrast/METHOD.md (AI Dataset Maker, Track C).", 8, DIM)
    pdf.output(path)
    return path


def markdown_pdf(md_path: str, out_path: str) -> str:
    """Minimal Markdown -> PDF: headings, paragraphs, lists, tables, code, blockquotes."""
    pdf = _pdf()
    lines = open(md_path, encoding="utf-8").read().splitlines()
    inline = lambda s: re.sub(r"\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|\[(.+?)\]\((.+?)\)",
                              lambda m: m.group(1) or m.group(2) or m.group(3) or m.group(4), s)
    k = 0
    while k < len(lines):
        ln = lines[k]
        if ln.startswith("```"):
            k += 1
            pdf.set_font("Plex", "", 7.5)
            pdf.set_fill_color(244, 241, 232)
            while k < len(lines) and not lines[k].startswith("```"):
                pdf.multi_cell(0, 3.6, lines[k] or " ", fill=True, new_x="LMARGIN", new_y="NEXT")
                k += 1
            pdf.ln(2)
        elif ln.startswith("|"):
            rows = []
            while k < len(lines) and lines[k].startswith("|"):
                cells = [inline(c.strip()) for c in lines[k].strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                k += 1
            k -= 1
            with pdf.table(text_align="LEFT", line_height=4.2, padding=1) as t:
                pdf.set_font("Plex", "", 8)
                for row in rows:
                    rr = t.row()
                    for c in row:
                        rr.cell(c)
            pdf.ln(2)
        elif ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#"))
            _h(pdf, inline(ln.lstrip("# ")), {1: 17, 2: 13, 3: 11}.get(lvl, 11))
        elif ln.startswith(("- ", "* ")):
            _p(pdf, "• " + inline(ln[2:]), 9.5)
        elif ln.startswith(">"):
            _p(pdf, inline(ln.lstrip("> ")), 9.5, DIM)
        elif ln.strip():
            para = [ln]
            while k + 1 < len(lines) and lines[k + 1].strip() and not lines[k + 1].startswith(("#", "|", "-", "*", ">", "```")):
                k += 1
                para.append(lines[k])
            _p(pdf, inline(" ".join(para)), 9.5)
            pdf.ln(1.5)
        k += 1
    pdf.output(out_path)
    return out_path


if __name__ == "__main__":
    out = markdown_pdf(os.path.join(ROOT, "contrast", "METHOD.md"), os.path.join(ROOT, "contrast", "METHOD.pdf"))
    from pypdf import PdfReader

    print(out, len(PdfReader(out).pages), "pages")

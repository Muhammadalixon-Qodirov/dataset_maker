"""Speech delivery dashboard (Track C) — baseline vs participant, grounded flaws, rubric.

    python contrast_app.py            # http://127.0.0.1:7861
    python contrast_app.py --share    # Colab
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contrast import compare as C
from contrast import rubric as R

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(ROOT, "contrast_data", "dataset")
SR = 16000

COLORS = {"rushed": "#D9480F", "dragging": "#1971C2", "long_pause": "#7048E8",
          "monotone": "#E8590C", "mumble": "#868E96", "stutter": "#C2255C"}


def _rows() -> list[dict]:
    p = os.path.join(DS, "metadata.csv")
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


ROWS = _rows()
SPEECHES = {}
for r in ROWS:
    if r["kind"] == "baseline":
        meta = json.load(open(os.path.join(DS, r["file"].replace(".flac", ".json")), encoding="utf-8"))
        SPEECHES[f"{meta['speaker']} — {meta['title']}"] = meta
_BASE_CACHE: dict[str, C.Aligned] = {}


def load(path: str) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=SR, mono=True)
    return wav.astype(np.float32)


def base_for(key: str, audio: str | None, text: str) -> C.Aligned:
    if audio:  # custom baseline
        return C.align(load(audio), text)
    if key not in _BASE_CACHE:
        meta = SPEECHES[key]
        _BASE_CACHE[key] = C.align(load(os.path.join(DS, meta["file"])), meta["transcript"])
    return _BASE_CACHE[key]


def warp(base_words, part_words):
    """Baseline time -> participant time, piecewise linear through word boundaries."""
    bx = np.array([t for w in base_words for t in (w.start, w.end)])
    px = np.array([t for w in part_words for t in (w.start, w.end)])
    order = np.argsort(bx)
    return lambda t: np.interp(t, bx[order], px[order])


def figure(res: dict, regions: list[dict]):
    fb, fp = res["base_frames"], res["part_frames"]
    wb, wp = res["base_words"], res["part_words"]
    f = warp(wb, wp)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Pitch (semitones re speaker median)", "Energy (dB re speaker median)",
                                        "Speech rate (syllables/s, 3-word window)"))
    tb = f(fb.t)
    for row, (yb, yp) in enumerate([(fb.pitch_st, fp.pitch_st), (fb.energy_db, fp.energy_db)], start=1):
        fig.add_trace(go.Scatter(x=tb, y=yb, name="baseline (time-warped)", mode="lines",
                                 line=dict(color="#ADB5BD", width=1.5), showlegend=row == 1,
                                 legendgroup="b"), row=row, col=1)
        fig.add_trace(go.Scatter(x=fp.t, y=yp, name="participant", mode="lines",
                                 line=dict(color="#1F7A52", width=1.8), showlegend=row == 1,
                                 legendgroup="p"), row=row, col=1)
    xm = [(w.start + w.end) / 2 for w in wp]
    fig.add_trace(go.Scatter(x=xm, y=[w.rate for w in wb], mode="lines+markers", name="baseline rate",
                             line=dict(color="#ADB5BD"), marker=dict(size=4), showlegend=False,
                             text=[w.word for w in wb], hovertemplate="%{text}: %{y:.1f} syl/s"), row=3, col=1)
    fig.add_trace(go.Scatter(x=xm, y=[w.rate for w in wp], mode="lines+markers", name="participant rate",
                             line=dict(color="#1F7A52"), marker=dict(size=4), showlegend=False,
                             text=[w.word for w in wp], hovertemplate="%{text}: %{y:.1f} syl/s"), row=3, col=1)
    for r in regions:
        for row in (1, 2, 3):
            fig.add_vrect(x0=r["start"], x1=max(r["end"], r["start"] + 0.08), fillcolor=COLORS[r["kind"]],
                          opacity=0.22, line_width=0, row=row, col=1)
        fig.add_annotation(x=(r["start"] + r["end"]) / 2, y=1.02, yref="paper", text=r["label"],
                           showarrow=False, font=dict(size=10, color=COLORS[r["kind"]]))
    fig.update_layout(height=640, margin=dict(l=50, r=20, t=60, b=40), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.08), plot_bgcolor="#FBFAF6")
    fig.update_xaxes(title_text="participant time (s)", row=3, col=1)
    return fig


def rubric_html(rub: dict) -> str:
    def color(v):
        return "#1F7A52" if v >= 8 else "#B7791F" if v >= 5 else "#C2410C"
    cards = "".join(
        f'<div class="rc"><b style="color:{color(v)}">{v:.1f}</b><span>{c}</span><i>{R.WHAT[c]}</i></div>'
        for c, v in rub["criteria"].items())
    return (f'<div class="rub"><div class="rc big"><b style="color:{color(rub["overall"])}">'
            f'{rub["overall"]:.1f}<small>/10</small></b><span>Overall delivery</span></div>{cards}</div>')


def analyze(speech, base_audio, base_text, part_choice, part_audio):
    text = base_text.strip() if base_audio else SPEECHES[speech]["transcript"]
    if base_audio and not text:
        raise gr.Error("Custom baseline needs its transcript.")
    if part_audio:
        ppath = part_audio
    elif part_choice:
        ppath = os.path.join(DS, part_choice)
    else:
        raise gr.Error("Record or upload a participant reading, or pick a file from the dataset.")
    base = base_for(speech, base_audio, text)
    try:
        part = C.align(load(ppath), text)
    except ValueError as e:
        raise gr.Error(f"Could not align the reading to the transcript ({e}). Read the same text.")
    res = C.compare(base, part)
    regions = C.as_dicts(res["regions"])
    rub = R.score(regions)
    table = [[f"{r['start']:.2f}–{r['end']:.2f}", r["label"], r["criterion"], r["words"][:50], r["explanation"]]
             for r in regions]
    gt_note = ""
    if not part_audio and part_choice:
        gt = json.load(open(os.path.join(DS, part_choice.replace(".flac", ".json")), encoding="utf-8"))["flaws"]
        if gt:
            gt_note = "**Ground truth (injected, hidden from the detector):** " + "; ".join(
                f"{g['type']} sev {g['severity']} at {g['start']:.2f}–{g['end']:.2f} s" for g in gt)
        else:
            gt_note = "**Ground truth:** clean baseline, no flaws injected."
    out = tempfile.NamedTemporaryFile(suffix=".json", delete=False, prefix="speech_report_")
    json.dump({"rubric": rub, "regions": regions, "participant": os.path.basename(ppath)}, open(out.name, "w"), indent=1)
    base_play = base_audio or os.path.join(DS, SPEECHES[speech]["file"])
    return rubric_html(rub), figure(res, regions), table, gt_note, ppath, base_play, out.name


def files_for(speech):
    sid = SPEECHES[speech]["id"]
    choices = [r["file"] for r in ROWS if r["speech"] == sid]
    return gr.update(choices=choices, value=f"{sid}/botched.flac"), SPEECHES[speech]["transcript"]


CSS = """
.rub{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.rc{border:1px solid var(--border-color-primary);border-radius:12px;padding:12px 14px;background:var(--background-fill-secondary)}
.rc b{display:block;font-size:30px;line-height:1.1}.rc small{font-size:14px;opacity:.6}
.rc span{display:block;font-weight:600;margin-top:2px}.rc i{display:block;font-size:12px;opacity:.7;font-style:normal}
.rc.big{grid-column:span 1;border-width:2px}
"""

INTRO = """
# 🎤 Contrastive Speech Analytics
**Where exactly did the delivery go wrong, and why?** Pick a reference speech (or upload your own),
then record or upload a reading of the **same text**. Both are force-aligned word by word; pitch,
energy, rate and pauses are compared on speaker-normalized scales, and every deviating region is
timestamped with its numeric cause. Built on the **AI Dataset Maker** alignment pipeline.
"""


def ui():
    first = next(iter(SPEECHES)) if SPEECHES else None
    sid = SPEECHES[first]["id"] if first else ""
    with gr.Blocks(title="Speech Flaw Grounding") as demo:
        gr.Markdown(INTRO)
        with gr.Tab("Analyze"):
            with gr.Row():
                with gr.Column(scale=1):
                    speech = gr.Dropdown(list(SPEECHES), value=first, label="1 · Reference (ideal) speech")
                    with gr.Accordion("…or upload your own reference", open=False):
                        base_audio = gr.Audio(type="filepath", label="Reference audio")
                        base_text = gr.Textbox(lines=4, label="Its transcript")
                    transcript = gr.Textbox(lines=6, label="Text to read", interactive=False,
                                            value=SPEECHES[first]["transcript"] if first else "")
                    part_audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                          label="2 · Record or upload a reading of this text")
                    part_choice = gr.Dropdown([r["file"] for r in ROWS if r["speech"] == sid],
                                              value=f"{sid}/botched.flac" if sid else None,
                                              label="…or pick a file from the contrastive dataset")
                    go_btn = gr.Button("Analyze delivery", variant="primary")
                with gr.Column(scale=2):
                    rub = gr.HTML()
                    plot = gr.Plot()
                    gt = gr.Markdown()
                    regions = gr.Dataframe(headers=["time (s)", "flaw", "criterion", "words", "why (numbers)"],
                                           wrap=True, interactive=False)
                    with gr.Row():
                        pa = gr.Audio(label="participant", type="filepath")
                        ba = gr.Audio(label="reference", type="filepath")
                    rep = gr.File(label="Report (JSON)")
            speech.change(files_for, speech, [part_choice, transcript])
            go_btn.click(analyze, [speech, base_audio, base_text, part_choice, part_audio],
                         [rub, plot, regions, gt, pa, ba, rep])
        with gr.Tab("Dataset & stress test"):
            p = os.path.join(ROOT, "contrast_data", "stress_report.md")
            gr.Markdown(open(p).read() if os.path.exists(p) else "Run `python -m contrast.stress_test`.")
            gr.Dataframe(value=[[r["file"], r["speaker"], r["kind"], r["severity"], r["flaw_types"], r["duration_s"]]
                                for r in ROWS],
                         headers=["file", "speaker", "kind", "severity", "flaws", "duration s"], interactive=False)
        with gr.Tab("Method"):
            gr.Markdown(open(os.path.join(ROOT, "contrast", "METHOD.md")).read()
                        if os.path.exists(os.path.join(ROOT, "contrast", "METHOD.md")) else "")
    return demo


if __name__ == "__main__":
    ui().queue(default_concurrency_limit=1).launch(
        server_port=7861, share="--share" in sys.argv, css=CSS, theme=gr.themes.Soft())

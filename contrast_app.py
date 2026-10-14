"""Speech delivery coach (Track C): one analysis feeds every panel.

    python contrast_app.py            # http://127.0.0.1:7861
    python contrast_app.py --share    # Colab

    compare()  ->  profile · flaw regions · word heat          (contrast/compare.py)
       │               │            │            │
       │           rubric.score (regions + profile, itemized)   (contrast/rubric.py)
       │               │            │
       │           coach: tips per region · profile findings · practice plan · A/B clips
       │               │
       └──────────►  report.analysis_pdf: everything above in one PDF
"""

from __future__ import annotations

import csv
import html
import json
import os
import sys
import tempfile

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contrast import coach as K
from contrast import compare as C
from contrast import report as P
from contrast import rubric as R

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(ROOT, "contrast_data", "dataset")
SR = 16000

COLORS = {"rushed": "#D9480F", "dragging": "#1971C2", "long_pause": "#7048E8", "monotone": "#E8590C",
          "mumble": "#868E96", "stutter": "#C2255C", "unclear": "#0B7285"}


def _rows() -> list[dict]:
    p = os.path.join(DS, "metadata.csv")
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


ROWS = _rows()
SPEECHES = {}
for r in ROWS:
    if r["kind"] == "baseline":
        meta = json.load(open(os.path.join(DS, r["file"].replace(".flac", ".json")), encoding="utf-8"))
        SPEECHES[f"{meta['speaker']} — {meta['title']}"] = meta
_BASE: dict[str, C.Aligned] = {}


def load(path: str) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=SR, mono=True)
    return wav.astype(np.float32)


def base_for(key: str, audio: str | None, text: str) -> C.Aligned:
    if audio:
        return C.align(load(audio), text)
    if key not in _BASE:
        meta = SPEECHES[key]
        _BASE[key] = C.align(load(os.path.join(DS, meta["file"])), meta["transcript"])
    return _BASE[key]


# ------------------------------------------------------------------ views (all read one result)


def warp(base_words, part_words):
    """Reference time -> participant time, piecewise linear through word boundaries."""
    bx = np.array([t for w in base_words for t in (w.start, w.end)])
    px = np.array([t for w in part_words for t in (w.start, w.end)])
    o = np.argsort(bx)
    return lambda t: np.interp(t, bx[o], px[o])


def figure(res: dict, regions: list[dict]):
    fb, fp, wb, wp = res["base_frames"], res["part_frames"], res["base_words"], res["part_words"]
    f = warp(wb, wp)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Pitch (semitones re own median)", "Energy (dB re own median)",
                                        "Speech rate (syllables/s, 3-word window)"))
    for row, (yb, yp) in enumerate([(fb.pitch_st, fp.pitch_st), (fb.energy_db, fp.energy_db)], start=1):
        fig.add_trace(go.Scatter(x=f(fb.t), y=yb, name="reference (time-warped)", mode="lines",
                                 line=dict(color="#ADB5BD", width=1.5), showlegend=row == 1, legendgroup="b"), row=row, col=1)
        fig.add_trace(go.Scatter(x=fp.t, y=yp, name="you", mode="lines", line=dict(color="#1F7A52", width=1.8),
                                 showlegend=row == 1, legendgroup="p"), row=row, col=1)
    xm = [(w.start + w.end) / 2 for w in wp]
    for ws, col in ((wb, "#ADB5BD"), (wp, "#1F7A52")):
        fig.add_trace(go.Scatter(x=xm, y=[w.rate for w in ws], mode="lines+markers", line=dict(color=col),
                                 marker=dict(size=4), showlegend=False, text=[w.word for w in wp],
                                 hovertemplate="%{text}: %{y:.1f} syl/s"), row=3, col=1)
    for k, r in enumerate(regions, 1):
        for row in (1, 2, 3):
            fig.add_vrect(x0=r["start"], x1=max(r["end"], r["start"] + 0.08), fillcolor=COLORS[r["kind"]],
                          opacity=0.2, line_width=0, row=row, col=1)
        fig.add_annotation(x=(r["start"] + r["end"]) / 2, y=1.02, yref="paper", text=f"#{k} {r['label']}",
                           showarrow=False, font=dict(size=10, color=COLORS[r["kind"]]))
    fig.update_layout(height=620, margin=dict(l=50, r=20, t=60, b=40), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.08), plot_bgcolor="#FBFAF6")
    fig.update_xaxes(title_text="your time (s)", row=3, col=1)
    return fig


def _col(v: float) -> str:
    return "#1F7A52" if v >= 8 else "#B7791F" if v >= 5 else "#C2410C"


def summary_html(rub: dict, plan: list[dict], findings: list[dict], profile: dict) -> str:
    cards = "".join(f'<div class="rc"><b style="color:{_col(v)}">{v:.1f}</b><span>{c}</span><i>{R.WHAT[c]}</i></div>'
                    for c, v in rub["criteria"].items())
    steps = "".join(f'<li><b>{html.escape(p["what"])}</b> <em>· {p["criterion"]}, ~{p["points"]:.1f} pts</em>'
                    f'<br>{html.escape(p["do"])}</li>' for p in plan) or "<li>Nothing to fix: delivery matches the reference.</li>"
    t, pr, g = profile["tempo_ratio"], profile["pause_ratio"], profile["pitch_range_ratio"]
    chips = (f'<span class="chip">tempo {(t - 1) * 100:+.0f}%</span>'
             f'<span class="chip">pauses ×{pr:.2f}</span>'
             f'<span class="chip">pitch range ×{g:.2f}</span>'
             f'<span class="chip">voice {profile["f0_median_hz"]["participant"]:.0f} Hz vs {profile["f0_median_hz"]["reference"]:.0f} Hz</span>')
    notes = "".join(f"<li>{html.escape(f['text'])}</li>" for f in findings)
    return f"""
<div class="rub"><div class="rc big"><b style="color:{_col(rub['overall'])}">{rub['overall']:.1f}<small>/10</small></b>
<span>Overall delivery</span></div>{cards}</div>
<div class="two">
 <div class="box"><h4>Practise these first</h4><ol>{steps}</ol></div>
 <div class="box"><h4>Your style vs the reference</h4><div>{chips}</div>
  <ul>{notes or '<li>Overall tempo, pauses and pitch range are close to the reference.</li>'}</ul>
  <p class="dim">Local flaws are measured against <b>your own</b> style, so a naturally slower or
  lower voice is not penalized everywhere.</p></div>
</div>"""


def heat_html(heat: list[dict]) -> str:
    def bg(h):
        return f"rgba(46,156,106,{0.10 + 0.25 * (1 - h):.2f})" if h < 0.35 else \
               f"rgba(227,178,87,{0.25 + 0.5 * h:.2f})" if h < 0.7 else f"rgba(240,138,75,{0.45 + 0.4 * h:.2f})"
    spans = []
    for w in heat:
        why = ", ".join(f"{d} {w[d]:.2f}" for d in C.HEAT_DIMS if w[d] >= 0.2)
        tip = (f"{w['start']:.2f}–{w['end']:.2f}s" + (f" · {C.KIND_LABEL[w['flaw']]}" if w["flaw"] else "")
               + (f" · {why}" if why else ""))
        spans.append(f'<span class="hw" style="background:{bg(w["heat"])}" title="{html.escape(tip)}">'
                     f'{html.escape(w["word"])}</span>')
    return ('<div class="heat">' + " ".join(spans) + '</div><p class="dim">Each word is colored by its largest '
            'deviation from the reference (pace, pause, pitch, volume, articulation). Hover a word for its numbers.</p>')


def analyze(speech, base_audio, base_text, part_choice, part_audio):
    text = base_text.strip() if base_audio else SPEECHES[speech]["transcript"]
    if base_audio and not text:
        raise gr.Error("A custom reference needs its transcript.")
    ppath = part_audio or (os.path.join(DS, part_choice) if part_choice else None)
    if not ppath:
        raise gr.Error("Record or upload a reading, or pick a file from the dataset.")
    base = base_for(speech, base_audio, text)
    try:
        part = C.align(load(ppath), text)
    except ValueError as e:
        raise gr.Error(f"Could not align the reading to the transcript ({e}). Please read the same text.")

    res = C.compare(base, part)                                   # ① ④ ⑤ one result
    regions = K.enrich(C.as_dicts(res["regions"]))                # ② tips
    rub = R.score(regions, res["profile"])                        # rubric: regions + profile
    findings = K.profile_findings(res["profile"])                 # ① style
    plan = K.practice_plan(regions, findings, rub)                # ② ranked by penalty

    table = [[f"#{k}", f"{r['start']:.2f}–{r['end']:.2f}", r["label"], r["words"][:40], r["explanation"], r["tip"]]
             for k, r in enumerate(regions, 1)]
    gt_note = ""
    if not part_audio and part_choice:
        gt = json.load(open(os.path.join(DS, part_choice.replace(".flac", ".json")), encoding="utf-8"))["flaws"]
        gt_note = ("**Ground truth (injected, hidden from the detector):** " + "; ".join(
            f"{g['type']} sev {g['severity']} at {g['start']:.2f}–{g['end']:.2f} s" for g in gt)) if gt \
            else "**Ground truth:** clean baseline, nothing injected."

    out = tempfile.mkdtemp(prefix="speech_")
    ref_name = speech if not base_audio else os.path.basename(base_audio)
    pdf = P.analysis_pdf(os.path.join(out, "delivery_report.pdf"), reference=ref_name,
                         participant=os.path.basename(ppath), rubric=rub, findings=findings, plan=plan,
                         heat=res["word_heat"], regions=regions)              # ⑥ report
    state = {"res": res, "regions": regions, "out": out}
    a, b = K.snippets(res, regions[0], out) if regions else (None, None)     # ③ first flaw ready
    return (summary_html(rub, plan, findings, res["profile"]), heat_html(res["word_heat"]),
            figure(res, regions), table, gt_note, a, b, pdf, state)


def listen(evt: gr.SelectData, state):
    """③ A/B: clicking a flaw row plays the same words, reference then you."""
    if not state or not state["regions"]:
        return None, None
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    r = state["regions"][min(row, len(state["regions"]) - 1)]
    return K.snippets(state["res"], r, state["out"])


def files_for(speech):
    sid = SPEECHES[speech]["id"]
    return (gr.update(choices=[r["file"] for r in ROWS if r["speech"] == sid], value=f"{sid}/botched.flac"),
            SPEECHES[speech]["transcript"])


CSS = """
.rub{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:12px}
.rc{border:1px solid var(--border-color-primary);border-radius:12px;padding:10px 12px;background:var(--background-fill-secondary)}
.rc b{display:block;font-size:28px;line-height:1.1}.rc small{font-size:13px;opacity:.6}
.rc span{display:block;font-weight:600}.rc i{display:block;font-size:12px;opacity:.7;font-style:normal}
.rc.big{border-width:2px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.box{border:1px solid var(--border-color-primary);border-radius:12px;padding:12px 14px}
.box h4{margin:0 0 6px}.box ol,.box ul{margin:4px 0 0 18px;padding:0}.box li{margin-bottom:6px}
.box em{opacity:.7;font-style:normal;font-size:13px}
.chip{display:inline-block;border:1px solid var(--border-color-primary);border-radius:99px;padding:2px 10px;margin:2px 4px 2px 0;font-size:13px}
.dim{opacity:.7;font-size:13px}
.heat{line-height:2.1;font-size:17px}.hw{padding:2px 4px;border-radius:5px;cursor:help}
"""

INTRO = """
# 🎤 Speech Delivery Coach: contrastive analytics & temporal flaw grounding
Pick a **reference speech** (or upload your own), then **record yourself** reading the same text, or pick a
file from the contrastive dataset. One analysis drives everything below: your **style profile**, every
**timestamped flaw** with its numeric cause and a **coaching tip**, a **word heatmap**, **A/B listening**
and a **PDF report**. Built on the AI Dataset Maker alignment pipeline.
"""


def ui():
    first = next(iter(SPEECHES)) if SPEECHES else None
    sid = SPEECHES[first]["id"] if first else ""
    with gr.Blocks(title="Speech Delivery Coach") as demo:
        gr.Markdown(INTRO)
        state = gr.State(None)
        with gr.Tab("Analyze"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    speech = gr.Dropdown(list(SPEECHES), value=first, label="1 · Reference (ideal) speech")
                    with gr.Accordion("…or upload your own reference", open=False):
                        base_audio = gr.Audio(type="filepath", label="Reference audio")
                        base_text = gr.Textbox(lines=4, label="Its transcript")
                    transcript_box = gr.Textbox(lines=6, label="Text to read", interactive=False,
                                                value=SPEECHES[first]["transcript"] if first else "")
                    part_audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                          label="2 · Record or upload your reading")
                    part_choice = gr.Dropdown([r["file"] for r in ROWS if r["speech"] == sid],
                                              value=f"{sid}/botched.flac" if sid else None,
                                              label="…or pick a file from the contrastive dataset")
                    go_btn = gr.Button("Analyze delivery", variant="primary")
                    pdf = gr.File(label="PDF report (everything on this page)")
                with gr.Column(scale=2):
                    summary = gr.HTML()
                    heat = gr.HTML()
                    plot = gr.Plot()
                    gt = gr.Markdown()
                    gr.Markdown("**Flaws.** Click a row to hear the same words: the reference first, then you.")
                    regions = gr.Dataframe(headers=["#", "time (s)", "flaw", "words", "why (numbers)", "what to do"],
                                           wrap=True, interactive=False)
                    with gr.Row():
                        ref_clip = gr.Audio(label="A · reference", type="filepath")
                        you_clip = gr.Audio(label="B · you", type="filepath")
            speech.change(files_for, speech, [part_choice, transcript_box])
            go_btn.click(analyze, [speech, base_audio, base_text, part_choice, part_audio],
                         [summary, heat, plot, regions, gt, ref_clip, you_clip, pdf, state])
            regions.select(listen, state, [ref_clip, you_clip])
        with gr.Tab("Dataset & stress test"):
            p = os.path.join(ROOT, "contrast_data", "stress_report.md")
            gr.Markdown(open(p).read() if os.path.exists(p) else "Run `python -m contrast.stress_test`.")
            gr.Dataframe(value=[[r["file"], r["speaker"], r["kind"], r["severity"], r["flaw_types"], r["duration_s"]]
                                for r in ROWS], headers=["file", "speaker", "kind", "severity", "flaws", "duration s"],
                         interactive=False)
        with gr.Tab("Method"):
            m = os.path.join(ROOT, "contrast", "METHOD.md")
            gr.Markdown(open(m).read() if os.path.exists(m) else "")
    return demo


if __name__ == "__main__":
    ui().queue(default_concurrency_limit=1).launch(
        server_port=7861, share="--share" in sys.argv, css=CSS, theme=gr.themes.Soft())

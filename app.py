"""AI Dataset Maker — web demo (Gradio). HF Spaces / Colab / lokal.

    python app.py            # http://127.0.0.1:7860
    python app.py --share    # Colab: ommaviy havola
"""

from __future__ import annotations

import csv
import html
import os
import re
import sys
import tempfile
import threading
import time
import zipfile
from queue import Empty, Queue

import gradio as gr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import pipeline as P
from src import quality as Q
from src import uz_text as T

ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_AUDIO = os.path.join(ROOT, "examples", "fleurs_uz_demo.flac")
EXAMPLE_TEXT = os.path.join(ROOT, "examples", "fleurs_uz_demo.txt")
# Bepul CPU'da (HF Spaces) Kotib ~1 s audio uchun ~1 s ishlaydi — demo uchun cheklov.
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "15"))

try:
    import torch

    HAS_GPU = torch.cuda.is_available()
except Exception:
    HAS_GPU = False

_asr: Q.KotibASR | None = None
_asr_lock = threading.Lock()


def get_asr() -> Q.KotibASR:
    global _asr
    with _asr_lock:
        if _asr is None:
            _asr = Q.KotibASR()
        return _asr


# ------------------------------------------------------------ reasons (UZ -> EN)

_REASONS = [
    (r"juda qisqa \(([\d.]+)s", r"too short (\1 s)"),
    (r"juda uzun \(([\d.]+)s", r"too long (\1 s)"),
    (r"ichki sukunat ([\d.]+)s", r"long pause inside (\1 s)"),
    (r"so'z sun'iy cho'zilgan \(([\d.]+)s", r"word stretched (\1 s) — alignment drift"),
    (r"alignment yiqilgan", "alignment failed"),
    (r"alignment ishonchi past \(mean ([\d.]+)\)", r"low alignment confidence (\1)"),
    (r"SNR ([\d.]+) dB", r"noisy (SNR \1 dB)"),
    (r"chegara CER (\d+)%", r"edge mismatch (\1%) — cut mid-word?"),
    (r"CER ([\d.]+)% — qo'lda tekshirish", r"ASR mismatch \1% — needs review"),
    (r"CER ([\d.]+)% > \d+%", r"ASR mismatch \1%"),
    (r"bo'sh bo'lak", "empty slice"),
]


def reason_en(r: str) -> str:
    r = re.sub(r"^(rad|ko'rish): ", "", r)
    for pat, rep in _REASONS:
        m = re.search(pat, r)
        if m:
            return m.expand(rep)
    return r


# ----------------------------------------------------------------- timeline SVG


def timeline_svg(wav_path: str, segments: list, accepted_starts: set[str], dur: float) -> str:
    """Waveform + kliplar: yashil = datasetga kirdi, qizil = rad etildi."""
    import soundfile as sf

    try:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:
        wav, sr = np.zeros(1, dtype=np.float32), 16000
    W, H, N = 1000, 120, 500
    step = max(1, wav.size // N)
    env = np.abs(wav[: step * N].reshape(-1, step)).max(axis=1) if wav.size >= N else np.abs(wav)
    env = env / (env.max() + 1e-9)
    mid = 70
    pts = " ".join(f"{i * W / max(1, len(env) - 1):.1f},{mid - v * 40:.1f}" for i, v in enumerate(env))
    pts2 = " ".join(f"{i * W / max(1, len(env) - 1):.1f},{mid + v * 40:.1f}" for i, v in reversed(list(enumerate(env))))
    bars = []
    for s in segments:
        x = s.start / dur * W
        w = max(1.0, (s.end - s.start) / dur * W)
        ok = f"{s.start:.3f}" in accepted_starts
        cls = "ok" if ok else "bad"
        tip = html.escape(f"{s.start:.1f}–{s.end:.1f}s · {s.text[:80]}")
        bars.append(f'<rect class="{cls}" x="{x:.1f}" y="6" width="{w:.1f}" height="14" rx="2"><title>{tip}</title></rect>')
    ticks = []
    for k in range(0, int(dur) + 1, max(10, int(dur / 10) // 10 * 10 or 10)):
        x = k / dur * W
        ticks.append(f'<text x="{x:.1f}" y="{H - 2}" class="t">{k // 60}:{k % 60:02d}</text>')
    return f"""
<div class="tl">
<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" width="100%" height="{H}">
  <style>
    .ok{{fill:#22a06b}} .bad{{fill:#e5484d}} .wf{{fill:#8b8d98;opacity:.55}}
    .t{{font:10px sans-serif;fill:#8b8d98}}
  </style>
  {''.join(bars)}
  <polygon class="wf" points="{pts} {pts2}"/>
  {''.join(ticks)}
</svg>
<div class="lg"><span class="sw ok"></span> kept in dataset &nbsp; <span class="sw bad"></span> rejected (hover a bar for its text)</div>
</div>"""


# ------------------------------------------------------------------- main run


def _read_csv(path: str) -> list[list[str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="|"))


def build(audio, text, text_file, asr_check, max_pause, sr):
    empty = (None, None, None, None, None, None, None)
    if not audio:
        raise gr.Error("Upload an audio file (or press “Load example”).")
    if text_file and not (text or "").strip():
        with open(text_file, encoding="utf-8") as f:
            text = f.read()
    if not (text or "").strip():
        raise gr.Error("Paste the transcript of the audio (or upload a .txt).")

    import soundfile as sf

    try:
        dur = sf.info(audio).duration
    except Exception:
        dur = 0.0
    if dur and dur > MAX_MINUTES * 60:
        raise gr.Error(f"This demo accepts up to {MAX_MINUTES:.0f} min of audio "
                       f"(got {dur/60:.1f} min). Use the CLI for full books.")

    out = tempfile.mkdtemp(prefix="dataset_")
    opt = P.Options(sr=int(sr), max_pause=float(max_pause), asr_check=bool(asr_check))
    q: Queue = Queue()
    box: dict = {}

    def worker():
        try:
            asr = get_asr() if asr_check else None
            box["report"] = P.run(audio, out, text=text, opt=opt, log=q.put, asr=asr)
        except Exception as e:  # foydalanuvchiga ko'rsatamiz
            box["error"] = e
        finally:
            q.put(None)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    lines: list[str] = []
    t0 = time.time()
    if asr_check and _asr is None:
        lines.append("loading Kotib ASR model (first run only, ~1.5 GB)...")
    while True:
        try:
            msg = q.get(timeout=1.0)
        except Empty:
            yield ("\n".join(lines) + f"\n… {time.time()-t0:.0f}s", *empty[1:])
            continue
        if msg is None:
            break
        lines.append(msg)
        yield ("\n".join(lines), *empty[1:])
    th.join()
    log_text = "\n".join(lines)
    if "error" in box:
        raise gr.Error(f"Pipeline failed: {box['error']}")

    rep = box["report"]
    segments = rep.pop("_segments")
    seg_rows = _read_csv(os.path.join(out, "segments.csv"))[1:]
    meta = {r[0]: r[1] for r in _read_csv(os.path.join(out, "metadata.csv")) if len(r) >= 2}
    by_start = {f"{s.start:.3f}": s for s in segments}

    kept = []
    for name, start, end, _ms, cer, _snr in seg_rows:
        s = by_start.get(start)
        kept.append([
            name, f"{float(start):.1f}", f"{float(end) - float(start):.1f}",
            meta.get(name, s.text if s else ""),
            f"{float(cer) * 100:.1f}%" if cer else "—",
            getattr(s, "hyp", "") if s else "",
        ])
    rejected = []
    for row in _read_csv(os.path.join(out, "rejected.csv"))[1:]:
        t, reason, _cer, _mean, txt = (row + [""] * 5)[:5]
        rejected.append([t.split("-")[0], reason_en(reason), txt])

    n_all = len(kept) + len(rejected)
    hours = rep.get("jami_soat", 0.0)
    cers = sorted(float(r[4]) for r in seg_rows if r[4])
    med = f"{np.median(cers) * 100:.1f}%" if cers else "n/a"
    zero = sum(1 for c in cers if c == 0)
    stats = f"""
<div class="cards">
  <div class="card"><b>{len(kept)}</b><span>clips kept</span></div>
  <div class="card"><b>{rep.get('saqlanish_foizi', 0):.0f}%</b><span>yield ({len(kept)}/{n_all} segments)</span></div>
  <div class="card"><b>{hours * 60:.1f} min</b><span>clean speech · avg {rep.get('ortacha_sek', 0)} s/clip</span></div>
  <div class="card"><b>{med}</b><span>median CER{f' · {zero} clips CER=0' if cers else ' (ASR check off)'}</span></div>
</div>"""

    zpath = os.path.join(out, "dataset_ljspeech.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in ("metadata.csv", "segments.csv", "rejected.csv", "report.json"):
            p = os.path.join(out, fn)
            if os.path.exists(p):
                z.write(p, f"dataset/{fn}")
        wdir = os.path.join(out, "wavs")
        for fn in sorted(os.listdir(wdir)):
            z.write(os.path.join(wdir, fn), f"dataset/wavs/{fn}")

    accepted_starts = {r[1] for r in seg_rows}
    accepted_starts = {f"{float(s):.3f}" for s in accepted_starts}
    svg = timeline_svg(audio, segments, accepted_starts, max(dur, segments[-1].end if segments else 1.0))
    first = os.path.join(out, "wavs", kept[0][0] + ".wav") if kept else None
    yield (log_text, stats + svg, kept, rejected, zpath, first, out)


def play(evt: gr.SelectData, table, out_dir):
    """Jadvaldagi qatorni bosganda — o'sha klipni ijro etish."""
    if not out_dir or table is None:
        return None
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    try:
        name = table.iloc[row, 0] if hasattr(table, "iloc") else table[row][0]
    except Exception:
        return None
    path = os.path.join(out_dir, "wavs", f"{name}.wav")
    return path if os.path.exists(path) else None


def load_example():
    with open(EXAMPLE_TEXT, encoding="utf-8") as f:
        return EXAMPLE_AUDIO, f.read()


def normalize_demo(text: str):
    ds = T.normalize_for_dataset(text)
    al = T.romanize_for_align(ds)
    return ds, al


# ------------------------------------------------------------------------- UI

CSS = """
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:4px 0 14px}
.card{border:1px solid var(--border-color-primary);border-radius:10px;padding:12px 14px;background:var(--background-fill-secondary)}
.card b{display:block;font-size:26px;line-height:1.2}
.card span{font-size:13px;opacity:.75}
.tl{border:1px solid var(--border-color-primary);border-radius:10px;padding:8px 10px}
.lg{font-size:12px;opacity:.8;margin-top:4px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle}
.sw.ok{background:#22a06b}.sw.bad{background:#e5484d}
#hero h1{margin-bottom:0}
"""

HERO = """
<div id="hero">
<h1>🎙️ AI Dataset Maker — Uzbek TTS</h1>
<p><b>Long Uzbek audio + its transcript → a clean, training-ready TTS dataset (LJSpeech).</b>
Forced alignment (MMS) cuts the recording into 2–12 s clips at sentence boundaries,
then every clip is re-transcribed by an Uzbek ASR (Kotib) and rejected if it does not
match its text. Tested on a 2-hour audiobook: <b>916 clips, 1.57 h, median CER 2.2%</b>.</p>
</div>
"""

RESULTS_MD = """
### Measured results

| Test | Input | Output | Yield | Median CER |
|---|---|---|---|---|
| FLEURS `uz_uz` (human transcripts) | 10.1 min, 977 words | 51 clips · 6.5 min | 75% | **2.11%** |
| *O'tkan kunlar* audiobook, 1 track | 9.6 min, 707 words | 72 clips · 7.6 min | 88% | **1.70%** |
| *O'tkan kunlar* audiobook, tracks 1–4 | **120 min**, 9 354 words | **916 clips · 1.57 h** | 88% | **2.22%** |

- Alignment: 977/977 words, **RTF 0.027** (10 min of audio aligned in 16 s on an RTX 3050).
- No drift over 2 hours: 240 / 218 / 219 / 239 clips per 30-minute quarter.
- Kept vs rejected clips separate cleanly: median CER **2.11% vs 14.7%** → the filter works.

### Yield vs published corpora

| Corpus | Kept |
|---|---|
| Libriheavy | ~83% |
| HiFiTTS-2 | 82% |
| **AI Dataset Maker** | **75–88%** |
| EuroSpeech | 65% |
| WenetSpeech4TTS | ~56% |
| Emilia | 29% |
"""

HOW_MD = """
### Pipeline

```
long audio ─┐
            ├─ 1. load         16 kHz mono
transcript ─┤
            ├─ 2. uz_text      Uzbek normalizer: numbers, money, dates, phones, abbreviations,
            │                  apostrophes (o' g' → oʻ gʻ), Roman-numeral guard
            ├─ 3. align        MMS_FA forced alignment, 20 s chunks; 5-min blocks for long books
            ├─ 4. segment      2–12 s at sentence boundaries, never mid-word,
            │                  silero-VAD edge snap, 4 filters (length, pause, stretched word…)
            ├─ 5. ASR check    Kotib (Uzbek Whisper) re-transcribes each clip → CER gate
            └─ 6. export       loudness −23 LUFS, symmetric padding, 22.05/24 kHz LJSpeech
```

### Why this is hard for Uzbek
- **Whisper fails on Uzbek** (WER 90%) → we use Kotib (CER 7%).
- **Text ≠ speech**: `1 250 000,50 so'm` must become *bir million ikki yuz ellik ming soʻm ellik tiyin*.
  Two bugs in existing Uzbek normalizer libraries were found and worked around.
- **Five apostrophe variants** (`'` `` ` `` `‘` `’` `ʻ`) for the same letter.
- **No public tool** does this end-to-end for Uzbek (see `XIZMATLAR.md`).

### Output
`wavs/0001.wav …` · `metadata.csv` (`0001|text`) · `segments.csv` (source timing + CER) ·
`rejected.csv` (with reasons) · `report.json` — ready for VITS, Piper, Coqui, StyleTTS2, F5-TTS.

CLI for full books: `python main.py --audio book.wav --text book.txt --out dataset/`
"""


# Gradio 6: theme/css launch() ga ko'chgan; 4.x/5.x: Blocks() da
GR6 = int(gr.__version__.split(".")[0]) >= 6
STYLE = {"css": CSS, "theme": gr.themes.Soft()}


def ui() -> gr.Blocks:
    hw = "GPU" if HAS_GPU else "CPU — ASR check is off by default (slow); use Colab GPU for the full pipeline"
    with gr.Blocks(title="AI Dataset Maker", **({} if GR6 else STYLE)) as demo:
        gr.HTML(HERO)
        with gr.Tab("Build dataset"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio = gr.Audio(label="Long audio (wav / mp3 / flac / ogg)", type="filepath")
                    text = gr.Textbox(label="Transcript (Uzbek, Latin script)", lines=8,
                                      placeholder="Paste the full text read in the audio…")
                    text_file = gr.File(label="…or upload transcript .txt", file_types=[".txt"],
                                        type="filepath")
                    with gr.Accordion("Options", open=False):
                        asr_check = gr.Checkbox(value=HAS_GPU, label="ASR back-check (Kotib) — strongest filter"
                                                + ("" if HAS_GPU else " · slow on CPU, several min"))
                        max_pause = gr.Slider(0.5, 2.0, value=0.93, step=0.01,
                                              label="Max pause inside a clip, s (0.93 plain reading · 1.5 dramatic)")
                        sr = gr.Dropdown(["22050", "24000"], value="22050",
                                         label="Output sample rate (22050 VITS/Piper · 24000 XTTS/StyleTTS2)")
                    with gr.Row():
                        ex_btn = gr.Button("Load example (FLEURS, 2 min)")
                        run_btn = gr.Button("Build dataset", variant="primary")
                    gr.Markdown(f"<small>Running on: {hw}. Demo limit {MAX_MINUTES:.0f} min.</small>")
                with gr.Column(scale=2):
                    stats = gr.HTML()
                    with gr.Tab("Kept clips"):
                        kept = gr.Dataframe(
                            headers=["clip", "start s", "dur s", "text", "CER", "ASR heard"],
                            wrap=True, interactive=False)
                        player = gr.Audio(label="Click a row to listen", type="filepath")
                    with gr.Tab("Rejected"):
                        rejected = gr.Dataframe(headers=["at s", "reason", "text"],
                                                wrap=True, interactive=False)
                    with gr.Tab("Log"):
                        log = gr.Textbox(lines=18, max_lines=30, show_label=False)
                    zfile = gr.File(label="Download dataset (LJSpeech zip)")
                    out_dir = gr.State(None)

            ex_btn.click(load_example, outputs=[audio, text])
            run_btn.click(build, [audio, text, text_file, asr_check, max_pause, sr],
                          [log, stats, kept, rejected, zfile, player, out_dir])
            kept.select(play, [kept, out_dir], player)

        with gr.Tab("Uzbek text normalizer"):
            gr.Markdown("TTS needs text exactly as spoken. Try numbers, money, times, phones, abbreviations.")
            src = gr.Textbox(label="Input", lines=3, value=(
                "2026-yil 10-sentyabrda prof. A. Qodirov soat 14:30 da 1 250 000,50 so'm "
                "to'ladi. Tel: +998 90 123 45 67. Masofa 25 km, O`zbekiston g'alabasi 15%."))
            btn = gr.Button("Normalize", variant="primary")
            ds = gr.Textbox(label="Dataset text (what goes into metadata.csv)", lines=3)
            al = gr.Textbox(label="Alignment text (what the aligner sees)", lines=3)
            btn.click(normalize_demo, src, [ds, al])

        with gr.Tab("Results"):
            gr.Markdown(RESULTS_MD)
        with gr.Tab("How it works"):
            gr.Markdown(HOW_MD)
    return demo


if __name__ == "__main__":
    ui().queue(default_concurrency_limit=1).launch(
        share="--share" in sys.argv, **(STYLE if GR6 else {}))

---
title: AI Dataset Maker — Uzbek TTS
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Long Uzbek audio + transcript → clean TTS dataset
---

# 🎤 Contrastive Speech Analytics & Temporal Flaw Grounding

**Multimodal AI Hackathon 2026 · Track C** · built on the AI Dataset Maker alignment pipeline

> Upload a speech and its transcript. The system tells you **where** the delivery deviates from a
> reference performance (exact start/end), **what** went wrong (pace, pauses, intonation, volume,
> fluency) and **why**, in numbers, plus a reproducible 0–10 rubric.

| Deliverable | Where |
|---|---|
| Codebase | [`contrast/`](contrast/) (pipeline) · [`contrast_app.py`](contrast_app.py) (dashboard) |
| Custom contrastive dataset | [`contrast_data/dataset/`](contrast_data/dataset/): 63 FLAC files + JSON labels + [`metadata.csv`](contrast_data/dataset/metadata.csv) |
| Interactive dashboard | `python contrast_app.py` → http://127.0.0.1:7861 (or the [Colab notebook](https://colab.research.google.com/github/Muhammadalixon-Qodirov/dataset_maker/blob/main/demo_colab.ipynb)) |
| Technical documentation | [`contrast/METHOD.pdf`](contrast/METHOD.pdf) (source: [`METHOD.md`](contrast/METHOD.md)): dataset construction, rubric, model, scoring |
| Stress-test results | [`contrast_data/stress_report.md`](contrast_data/stress_report.md) |

### Quick start

**Prerequisites:** Python 3.10–3.13, ~3 GB disk (the MMS aligner model, 1.2 GB, downloads on first
run), CPU is enough (a GPU is optional).

```bash
git clone https://github.com/Muhammadalixon-Qodirov/dataset_maker.git && cd dataset_maker
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python contrast_app.py                 # dashboard: pick a speech, record yourself, analyze
python -m contrast.stress_test         # reproduce the metrics below (~10 min on CPU)
python -m contrast.build_dataset       # rebuild the dataset from the public-domain originals
```

### How it works

```
reference speech ─┐                     ┌─ F0 (semitones re speaker median)   ─┐
                  ├─ MMS forced align ──┼─ energy (dB re speaker median)        ├─ word-by-word deltas
participant  ─────┘  same transcript,   ├─ articulation rate, pauses           │  + z vs pair noise
                     wildcard * between ├─ spectral centroid, HNR, MFCC        │        │
                     words              └─ extra sound outside the text (*)   ─┘        ▼
                                                              flaw regions [start, end] + numeric cause
                                                                       │
                                                       rubric: Pace · Pausing · Expressiveness ·
                                                               Volume & clarity · Fluency (0–10)
```

**Dataset.** Three public-domain speeches (FDR 1941, JFK 1962, Reagan 1986) are the "good"
baselines. Each is mirrored by **6 flaw types × 3 severities** (rushed, dragging, long pause,
monotone, mumble, stutter), plus a *near-perfect* and a *botched* composite: 63 files. Flaws are
injected by signal processing (PSOLA time-scaling with pitch preserved, Praat pitch-tier
flattening, room-tone pauses, band-pass + gain, repeated onsets). The ground-truth timestamps
are therefore **exact to the sample**: each operation carries a time map, so no hand-labeling
error enters the dataset.

**Detector.** Both readings are aligned to the same words, so word *k* in one is word *k* in the
other. Deviations are reported in natural units (syl/s, s, semitones, dB) with a z-score
against the pair's own measurement noise, merged into regions and explained in plain English:

> **3.96–4.94 s · Rushed pace.** Speech rate 9.2 syl/s vs baseline 6.2 syl/s (+47%, z=+11.2).
> Words are compressed; listeners lose the content.

### One analysis, every view

Everything comes from a **single word-aligned delta table**, so the panels always agree:

| View | Built from |
|---|---|
| **Style profile**: overall tempo, pauses, pitch range vs the reference | medians of the deltas; local flaws are judged *against your own style*, so a slower or lower voice isn't flagged everywhere |
| **Timestamped flaws** with numeric causes | deviations from the profile, z-scored against the pair's noise; 7 kinds incl. MFCC-DTW articulation |
| **Rubric** (5 criteria, 0–10, deterministic) | region penalties + profile habits, itemized |
| **Coaching tip per flaw** | that region's numbers ("aim for 6.3 syl/s, you were at 9.2") |
| **Practice plan**: top 3 | the rubric items that cost the most points |
| **Word heatmap** | the same per-word deltas, 0–1 |
| **A/B listening** | the region's word span cut from both recordings |
| **PDF report** | all of the above in one file |

### Results (stress test: the detector never sees the labels)

63 files · recall **81%** (severity 1: 71%, 2: 89%, 3: 83%) · precision **66%** · false alarms on clean baselines: **0**

| Flaw | sev 1 | sev 2 | sev 3 | start err (median) | end err (median) | score s1 → s2 → s3 |
|---|---|---|---|---|---|---|
| dragging | 3/3 | 3/3 | 2/3 | 0.078 s | 0.273 s | 9.47 → 9.03 → 8.8 |
| long_pause | 3/3 | 3/3 | 5/6 | 0.032 s | 0.027 s | 9.37 → 9.13 → 8.37 |
| monotone | 5/6 | 3/3 | 6/6 | 0.335 s | 0.383 s | 9.5 → 8.97 → 8.73 |
| mumble | 3/3 | 3/3 | 3/3 | 0.119 s | 0.021 s | 8.93 → 8.27 → 7.17 |
| rushed | 3/6 | 3/3 | 6/6 | 0.051 s | 0.02 s | 9.77 → 9.33 → 9.1 |
| stutter | 0/3 | 1/3 | 3/6 | 0.045 s | 0.226 s | 9.87 → 9.9 → 9.6 |

Rubric falls monotonically with severity for 5/6 flaw types.
Composite files: {'baseline': 10.0, 'near_perfect': 9.0, 'botched': 5.33}
**Reading the table.** Rows are flaw types and columns are severities (counts are found/injected).
A flaw is "found" when a predicted region of the **same type** overlaps its injected timestamps
(IoU ≥ 0.2; any overlap for point-like pauses and stutters). Boundary errors are medians over
found flaws. The last column is the mean overall rubric score at each severity.

**What works.** Zero false alarms on the three clean baselines. Pauses, pace and volume are
localized within tens of milliseconds. Severity 2–3 flaws are found 83–89% of the time. The rubric
falls with severity for 5 of 6 flaw types, and from 10.0 (baseline) to 9.0 (near-perfect) to
5.3 (botched).

**What doesn't (yet).** *Stutter* is the weak class (4/12): a single repeated onset is short,
and the aligner often folds it into the neighbouring word, where it reads as slow pace. Subtle
severity-1 *rushed* readings (×0.8) are found 3/6 times. Precision (66%) is lowered mostly by
extra regions in the *botched* composites, where four overlapping flaws shift the alignment.

**Articulation probe.** The synthetic set contains no mispronunciations, so we reverse one content word per position (same energy and spectrum, wrong sound order; 3 per speech). **4/9** are reported as *unclear articulation* at the right word and **7/9** are flagged as some flaw at that spot; 2/9 are missed. This is the least mature detector.

### AI tools disclosure

As the hackathon rules require, this is the AI assistance used in the project.
**Claude Code (Anthropic)** was used as a coding assistant throughout Track C: writing and
refactoring the `contrast/` modules, the dashboard (`contrast_app.py`), the dataset builder and
stress test, debugging detector thresholds against the stress-test output, drafting the
documentation (`contrast/METHOD.md`), and producing the demo video. In Module 1 (Uzbek TTS) it was
used for the Gradio web demo (`app.py`), the shared `src/pipeline.py` refactor, the English README and the
Colab notebook; the original pipeline (`src/`, `main.py`) and its research were the author's work. The demo video was
recorded automatically from the running dashboard (Playwright), and its English narration is
synthetic (Piper TTS, `en_US-ryan-high`). The project idea, direction and the reuse of the
author's own AI Dataset Maker pipeline (alignment, VAD, dataset construction) come from the
author, who reviewed the results and is responsible for the submission. Third-party models
used at runtime: Meta MMS_FA (alignment), Praat via parselmouth (pitch/intensity), librosa (MFCC).

---

# 🎙️ Module 1: AI Dataset Maker for Uzbek TTS

**Turn a long Uzbek recording (an audiobook, a lecture, a podcast) and its text into a clean,
training-ready text-to-speech dataset — automatically.**

```bash
python main.py --audio book.mp3 --text book.txt --out dataset/
```

Output: LJSpeech format (`wavs/` + `metadata.csv`), ready for VITS, Piper, Coqui, StyleTTS2, F5-TTS.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Muhammadalixon-Qodirov/dataset_maker/blob/main/demo_colab.ipynb)

[🤗 Results page](https://huggingface.co/spaces/muhammadalixon/ai-dataset-maker) · [▶ Live demo](#demo) · [Results](#results) · [How it works](#how-it-works) · [O'zbekcha](README_UZ.md)

---

## The problem

Uzbek has **~35 million speakers** but almost no open, high-quality TTS data. Building one by
hand means listening to hours of audio and cutting it into thousands of sentence clips, each with
exactly matching text. That takes **weeks of manual work per hour of speech**.

Hours of Uzbek audiobooks with published text already exist. What's missing is a tool that turns
them into a dataset. Generic tools fail on Uzbek:

- **Whisper does not work for Uzbek** (WER 90%), so you can't just transcribe and cut.
- **Text ≠ speech.** `1 250 000,50 so'm` is *read* as *bir million ikki yuz ellik ming soʻm ellik tiyin*.
  A TTS model trained on the digits learns nothing.
- **Five apostrophe variants** (`'` `` ` `` `‘` `’` `ʻ`) for the same letters oʻ / gʻ.
- Existing pipelines (NeMo SDP, Emilia, Libriheavy) target English or Chinese. **None handles Uzbek end-to-end.**

## The solution

A 6-stage pipeline. Every clip is **verified by a second model** before it reaches the dataset:

```
long audio ─┐
            ├─ 1. load         16 kHz mono, any format
transcript ─┤
            ├─ 2. uz_text      Uzbek normalizer: numbers, money, dates, times, phones,
            │                  units, abbreviations, apostrophes, Roman-numeral guard
            ├─ 3. align        MMS forced alignment → timestamp of every word
            │                  (5-min blocks for multi-hour books, no drift)
            ├─ 4. segment      2–12 s clips at sentence boundaries, never mid-word,
            │                  silero-VAD edge snap, 4 quality filters
            ├─ 5. ASR check    Kotib (Uzbek Whisper) re-listens to each clip;
            │                  CER > 10% → rejected or flagged for review
            └─ 6. export       −23 LUFS loudness, symmetric padding, 22.05 / 24 kHz
```

Nothing is thrown away silently: `rejected.csv` lists every rejected clip with its reason.

## Results

Measured against **human-written ground truth**:

| Test | Input | Dataset produced | Kept | Median CER |
|---|---|---|---|---|
| FLEURS `uz_uz` | 10 min, 977 words | 51 clips · 6.5 min | 75% | **2.11%** |
| *O'tkan kunlar* audiobook | 9.6 min, 707 words | 72 clips · 7.6 min | 88% | **1.70%** |
| *O'tkan kunlar* audiobook, 4 tracks | **2 hours**, 9 354 words | **916 clips · 1.57 h** | 88% | **2.22%** |

- **977 / 977** words aligned. Alignment speed **RTF 0.027** (10 min of audio in 16 s, RTX 3050).
- **No drift over 2 hours**: 240 / 218 / 219 / 239 clips in each 30-minute quarter.
- The filter separates clips cleanly: median CER **2.1% kept vs 14.7% rejected**.
- Hallucinations are caught: alignment scores on hallucinated words are ~40× lower than on real ones.

**Yield compared with published corpora:**

| Corpus | Kept |
|---|---|
| Libriheavy | ~83% |
| HiFiTTS-2 | 82% |
| **AI Dataset Maker** | **75–88%** |
| EuroSpeech | 65% |
| WenetSpeech4TTS | ~56% |
| Emilia | 29% |

Full report (Uzbek): [NATIJALAR.md](NATIJALAR.md).

## Demo

**Web UI** (Gradio): upload audio and text, see every clip on a timeline (green = kept,
red = rejected), listen to clips, and download the dataset as a zip. It also has a live
**Uzbek text normalizer** tab.

| Where | How |
|---|---|
| Google Colab (free GPU) | [open in Colab](https://colab.research.google.com/github/Muhammadalixon-Qodirov/dataset_maker/blob/main/demo_colab.ipynb) → *Runtime → Run all* |
| Hugging Face Space | [results page](https://huggingface.co/spaces/muhammadalixon/ai-dataset-maker): a real ASR-checked run, listen to every kept and rejected clip, no install |
| Local | `pip install -r requirements.txt gradio && python app.py` |

The **Load example** button runs a 2-minute FLEURS sample (CC BY 4.0) with human transcripts.

## Install

```bash
# GPU (CUDA 12.4); on CPU skip the first line
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Models download automatically on first run: MMS_FA (1.2 GB) and Kotib `Kotib/uzbek_stt_v1`.
An 8 GB GPU is enough. CPU also works, just more slowly.

## CLI

```bash
python main.py --audio book.mp3 --text book.txt --out dataset/
```

| Flag | Default | Meaning |
|---|---|---|
| `--max-pause` | 0.93 | longest pause allowed inside a clip, s (`1.5` for dramatic reading) |
| `--sr` | 22050 | 22050 = VITS/Piper · 24000 = XTTS/StyleTTS2 |
| `--no-asr-check` | off | skip ASR verification (10× faster, weaker filter) |
| `--asr-text` | off | no transcript: let Kotib write one (lower quality) |
| `--batch` | 4 | ASR batch size; halves automatically on GPU OOM |

Long jobs are **resumable**: alignment and ASR results are cached in the output folder, so
an interrupted run continues where it stopped.

**Output**

| File | Contents |
|---|---|
| `wavs/0001.wav …` | clips, 16-bit PCM |
| `metadata.csv` | `0001\|text` (LJSpeech) |
| `segments.csv` | source start/end, alignment score, CER, SNR for each clip |
| `rejected.csv` | every rejected segment and why |
| `report.json` | statistics |

## Project layout

```
main.py            CLI
app.py             web demo (Gradio)
src/pipeline.py    the 6-stage pipeline (shared by CLI and demo)
src/uz_text.py     Uzbek text normalization
src/align.py       MMS_FA forced alignment, block mode for long audio
src/segment.py     clip boundaries + filters
src/vad.py         silero-VAD boundary snapping
src/quality.py     Kotib ASR back-check, CER, SNR, verdict
src/export.py      LJSpeech export, loudness, padding
tests/             FLEURS / audiobook evaluation scripts
tadqiqot/          research notes: 362+ sources on alignment, corpora, quality (Uzbek)
```

## Limitations

- The source should be **one clean voice**. Dialogue, music and heavy noise are rejected,
  which is the intended behavior.
- Plan on **~2 hours of recording for 1 hour of clean data**. Professional corpora discard
  40–60% of their input too.
- Thresholds were calibrated on two sources (FLEURS and one audiobook narrator).
- Training a TTS model on the output is the next step and has not been done yet.

## Vision

Uzbek speech is step one. The goal is a tool that produces **correct training data for any AI model**:

| Stage | Status |
|---|---|
| Uzbek speech (TTS) datasets from audiobooks | ✅ working, tested on 2 hours |
| Any language: the MMS aligner already covers 1,130 languages; each new one needs a text normalizer and an ASR | next |
| Other model types: ASR, TTS and LLM datasets, cleaned, verified and exported in the right format | later |

The principle stays the same everywhere: **nothing reaches the dataset unless a second model has checked it.**
We started with Uzbek on purpose. It is one of the hardest cases: Whisper fails on it and open data barely exists.

## Roadmap

Faster ASR check (faster-whisper / CTranslate2) · denoising stage · review UI for flagged
clips · multi-file batch mode · DNSMOS/UTMOS scoring · training a public Uzbek TTS voice.

## Credits

[MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) forced aligner (Meta) ·
[Kotib](https://huggingface.co/Kotib/uzbek_stt_v1) Uzbek STT ·
[silero-vad](https://github.com/snakers4/silero-vad) ·
[uztextnorm](https://pypi.org/project/uztextnorm/) · [uztext](https://github.com/abafaboy/uztext) ·
[FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0) for evaluation.

## License

Code: [MIT](LICENSE). Models are downloaded at runtime under their own licenses:
MMS_FA weights are **CC BY-NC 4.0** (non-commercial), Kotib is Apache-2.0, silero-vad is MIT.
For commercial use, swap the aligner for a commercially licensed one.

Author: **Muhammadalixon Qodirov**

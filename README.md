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

# 🎙️ AI Dataset Maker — Uzbek TTS

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

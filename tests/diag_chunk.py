"""Chunking siljish beradimi? Bitta klipni yolg'iz va uzun fayl ichida solishtirish.

1. Birinchi FLEURS klipni YOLG'IZ olib (chunk'siz, bir o'tishda) align qilamiz
2. Birinchi so'z qayerda chiqishini haqiqiy nutq boshlanishi bilan solishtiramiz
3. Shu klipni uzun fayl ichida (chunk bilan) align qilib, farqni ko'ramiz
"""

import io
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.align import SAMPLE_RATE, ForcedAligner, add_stars, strip_stars
from src.uz_text import split_aligned

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleurs_test")
truth = json.load(io.open(os.path.join(B, "truth.json"), encoding="utf-8"))
wav, sr = sf.read(os.path.join(B, "long.wav"), dtype="float32")
assert sr == SAMPLE_RATE

t0 = truth[0]
a, b = int(t0["start"] * sr), int(t0["end"] * sr)
clip = wav[a:b]

# haqiqiy nutq boshlanishi (klip ichida)
th = float(np.abs(clip).max()) * 0.05
nz = np.where(np.abs(clip) > th)[0]
speech_on = nz[0] / sr
speech_off = nz[-1] / sr
print(f"klip: {t0['start']:.2f}-{t0['end']:.2f}  ({len(clip)/sr:.2f}s)")
print(f"haqiqiy nutq: {speech_on:.3f} -> {speech_off:.3f} (klip ichida)")
print(f"matn: {t0['text'][:70]}")
print()

pairs = split_aligned(t0["text"])
words = [al for _, al in pairs]
al = ForcedAligner()

# ---- 1) YOLG'IZ, chunk'siz -------------------------------------------
wt = torch.from_numpy(clip)
emis = al.emissions(wt, chunk_sec=60.0, overlap_sec=2.0)  # klip 60s dan qisqa -> bir o'tish
sp1 = strip_stars(al.align(emis, add_stars(words, "edges")))
print(f"[A] YOLG'IZ (chunk'siz): {len(sp1)}/{len(words)} so'z")
if sp1:
    print(f"    1-so'z: {sp1[0].start:.3f}s  (nutq {speech_on:.3f}s) "
          f"-> farq {(sp1[0].start-speech_on)*1000:+.0f}ms")
    print(f"    oxirgi tugashi: {sp1[-1].end:.3f}s (nutq {speech_off:.3f}s) "
          f"-> farq {(sp1[-1].end-speech_off)*1000:+.0f}ms")
    print(f"    score median: {sorted(s.score for s in sp1)[len(sp1)//2]:.3f}")

# ---- 2) UZUN fayl ichida, chunk BILAN ---------------------------------
full_pairs = split_aligned(io.open(os.path.join(B, "text.txt"), encoding="utf-8").read())
full_words = [x for _, x in full_pairs]
wf = torch.from_numpy(wav)
emis2 = al.emissions(wf, chunk_sec=20.0, overlap_sec=2.0)
sp2 = strip_stars(al.align(emis2, add_stars(full_words, "edges")))
print(f"\n[B] UZUN fayl, chunk=20s: {len(sp2)}/{len(full_words)} so'z")
if sp2:
    abs_on = t0["start"] + speech_on
    print(f"    1-so'z: {sp2[0].start:.3f}s  (nutq {abs_on:.3f}s) "
          f"-> farq {(sp2[0].start-abs_on)*1000:+.0f}ms")
    print(f"    oxirgi so'z tugashi: {sp2[-1].end:.2f}s / audio {len(wav)/sr:.2f}s")
    print(f"    score median: {sorted(s.score for s in sp2)[len(sp2)//2]:.3f}")

# ---- 3) chunk hajmini o'zgartirib siljish bormi ------------------------
print("\n[C] chunk hajmi ta'siri (1-so'z boshlanishi):")
for cs in (10.0, 20.0, 40.0):
    e = al.emissions(wf, chunk_sec=cs, overlap_sec=2.0)
    s = strip_stars(al.align(e, add_stars(full_words, "edges")))
    print(f"    chunk={cs:4.0f}s -> 1-so'z {s[0].start:.3f}s | "
          f"oxirgi {s[-1].end:.2f}s | fps {al._fps:.2f}")

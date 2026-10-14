"""Demo misoli: FLEURS uz_uz dev kliplarini ulab ~2 daqiqalik "uzun" audio yasaydi.

FLEURS — CC BY 4.0, matnlar qo'lda yozilgan (ground truth).

    curl -L -o dev.tsv    https://huggingface.co/datasets/google/fleurs/resolve/main/data/uz_uz/dev.tsv
    curl -L -o dev.tar.gz https://huggingface.co/datasets/google/fleurs/resolve/main/data/uz_uz/audio/dev.tar.gz
    python examples/make_example.py dev.tsv dev.tar.gz
"""

import os
import sys
import tarfile

import numpy as np
import soundfile as sf

SR = 16000
GAP = 0.6  # kliplar orasidagi jimlik (MERGE_GAP 0.55 dan katta)
TARGET_SEC = 120.0
MIN_DUR, MAX_DUR = 3.0, 12.0
GENDER = "FEMALE"  # bitta ovozga yaqinroq bo'lishi uchun

OUT = os.path.dirname(os.path.abspath(__file__))
tsv_path, tar_path = sys.argv[1], sys.argv[2]

rows, seen = [], set()
for line in open(tsv_path, encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    if len(f) < 7 or f[6] != GENDER or f[2] in seen:
        continue
    dur = int(f[5]) / SR
    if MIN_DUR <= dur <= MAX_DUR:
        seen.add(f[2])
        rows.append((f[1], f[2], dur))

chosen, total = [], 0.0
for r in rows:
    if total >= TARGET_SEC:
        break
    chosen.append(r)
    total += r[2] + GAP
need = {r[0] for r in chosen}

audio: dict[str, np.ndarray] = {}
with tarfile.open(tar_path) as tar:
    for m in tar:
        name = os.path.basename(m.name)
        if name in need:
            wav, sr = sf.read(tar.extractfile(m), dtype="float32")
            assert sr == SR, sr
            audio[name] = wav

gap = np.zeros(int(GAP * SR), dtype=np.float32)
parts = [gap]
for name, _, _ in chosen:
    parts += [audio[name], gap]
wav = np.concatenate(parts)
sf.write(os.path.join(OUT, "fleurs_uz_demo.flac"), wav, SR)
with open(os.path.join(OUT, "fleurs_uz_demo.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(t for _, t, _ in chosen) + "\n")
print(f"{len(chosen)} klip, {wav.size / SR:.1f} s -> examples/fleurs_uz_demo.flac")

"""FLEURS uz_uz dan ~10 daqiqalik sinov materiali yasash.

Klipларni ulab uzun audio qiladi va HAQIQIY chegaralarni yozib oladi.
Shu tufayli alignment xatosini millisekundda o'lchash mumkin bo'ladi.
"""

import io
import json
import os
import tarfile

import numpy as np
import soundfile as sf

HF = "D:/hf_cache/hub/datasets--google--fleurs/snapshots"
OUT = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(OUT, "fleurs_test")
os.makedirs(TEST, exist_ok=True)

SR = 16000
GAP = 0.6          # kliplar orasidagi jimlik (MERGE_GAP 0.55 dan katta)
TARGET_MIN = 10.0  # daqiqa
MIN_DUR, MAX_DUR = 3.0, 12.0   # bitta klip = bitta segment bo'lishi uchun
GENDER = "FEMALE"  # bir xillik uchun

snap = os.path.join(HF, os.listdir(HF)[0], "data", "uz_uz")
tsv_path = os.path.join(snap, "dev.tsv")
tar_path = os.path.join(snap, "audio", "dev.tar.gz")

# ---------------------------------------------------- 1. TSV o'qish
rows = []
for line in io.open(tsv_path, encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    if len(f) < 7:
        continue
    dur = int(f[5]) / SR
    rows.append({"file": f[1], "text": f[2], "dur": dur, "gender": f[6]})
print(f"[1] TSV: {len(rows)} qator")

sel = [r for r in rows if r["gender"] == GENDER and MIN_DUR <= r["dur"] <= MAX_DUR]
print(f"[1] {GENDER}, {MIN_DUR}-{MAX_DUR}s: {len(sel)} klip")

# takrorlanmaydigan matnlar (FLEURS'da bir gap bir necha spiker tomonidan o'qilgan)
seen, uniq = set(), []
for r in sel:
    if r["text"] in seen:
        continue
    seen.add(r["text"])
    uniq.append(r)
print(f"[1] unikal matn: {len(uniq)}")

chosen, total = [], 0.0
for r in uniq:
    if total >= TARGET_MIN * 60:
        break
    chosen.append(r)
    total += r["dur"] + GAP
print(f"[1] tanlandi: {len(chosen)} klip, ~{total/60:.1f} daqiqa")

# ---------------------------------------------------- 2. audio ulash
want = {r["file"] for r in chosen}
audio: dict[str, np.ndarray] = {}
with tarfile.open(tar_path, "r:gz") as tf:
    for m in tf:
        base = os.path.basename(m.name)
        if base in want and m.isfile():
            data, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            assert sr == SR, f"kutilmagan sr: {sr}"
            audio[base] = data
            if len(audio) == len(want):
                break
print(f"[2] tardan o'qildi: {len(audio)}/{len(want)}")

gap_arr = np.zeros(int(GAP * SR), dtype=np.float32)
pieces, truth, cursor = [], [], 0.0
for r in chosen:
    a = audio.get(r["file"])
    if a is None:
        continue
    start = cursor
    pieces.append(a)
    cursor += len(a) / SR
    truth.append({"start": round(start, 3), "end": round(cursor, 3), "text": r["text"]})
    pieces.append(gap_arr)
    cursor += GAP

long_wav = np.concatenate(pieces)
wav_path = os.path.join(TEST, "long.wav")
sf.write(wav_path, long_wav, SR, subtype="PCM_16")
print(f"[2] yozildi: {wav_path}  ({len(long_wav)/SR/60:.2f} daqiqa)")

# ---------------------------------------------------- 3. matn va haqiqat
text = " ".join(t["text"] for t in truth)
txt_path = os.path.join(TEST, "text.txt")
io.open(txt_path, "w", encoding="utf-8").write(text)
print(f"[3] matn: {txt_path}  ({len(text)} belgi, {len(text.split())} so'z)")

json.dump(truth, io.open(os.path.join(TEST, "truth.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"[3] haqiqat: {len(truth)} segment chegarasi saqlandi")
print(f"[3] o'rtacha klip: {sum(t['end']-t['start'] for t in truth)/len(truth):.2f}s")

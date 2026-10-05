"""USC (Uzbek Speech Corpus, CC-BY-4.0) dan BITTA spikerning uzluksiz o'qishini yig'ish.

USC id formati: {spiker}_{sessiya}_{gap_raqami}_{n}. Bitta spikerning gaplari
ketma-ket adabiy matn — qaytadan ulasak audiokitob o'qishiga o'xshaydi.

Chiqish (tests/usc_test/):
    long.wav     — bitta spiker, ~TARGET_MIN daqiqa, kliplar orasida GAP jimlik
    text.txt     — ketma-ket matn (USC allaqachon normalizatsiyalangan: kichik harf, punktuatsiyasiz)
    truth.json   — har klipning haqiqiy chegarasi (baholash uchun)
    namuna.md    — foydalanuvchi ko'rib chiqishi uchun: spiker, gaplar, davomiylik
"""

import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "murodbek/uzbek-speech-corpus"
SHARDS = ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]
TARGET_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
GAP = 0.5  # kliplar orasidagi jimlik (sek) — tabiiy gap pauzasi
SR = 16000

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usc_test")
os.makedirs(OUT, exist_ok=True)


def speaker_of(uid: str) -> str:
    return uid.split("_")[0]


def order_key(uid: str):
    parts = uid.split("_")
    try:
        return (int(parts[1]), int(parts[2]), int(parts[3]) if len(parts) > 3 else 0)
    except (IndexError, ValueError):
        return (0, 0, 0)


# ------------------------------------------------ 1. shardlarni o'qish
rows = []
for sh in SHARDS:
    p = hf_hub_download(REPO, sh, repo_type="dataset")
    t = pq.read_table(p, columns=["id", "sentence"])
    rows.extend(t.select(["id", "sentence"]).to_pylist())
    print(f"[1] {sh}: {t.num_rows} qator")
print(f"[1] jami {len(rows)} klip")

# ------------------------------------------------ 2. spikerlar bo'yicha guruhlash
by_spk = defaultdict(list)
for r in rows:
    by_spk[speaker_of(r["id"])].append(r)

# davomiylikni bilish uchun audio kerak — avval klip soni bo'yicha top-10 ni olamiz
top = sorted(by_spk.items(), key=lambda kv: -len(kv[1]))[:10]
print(f"[2] spikerlar: {len(by_spk)} | eng ko'p klipli 10 tasi:")
for spk, lst in top:
    print(f"    {spk}: {len(lst)} klip")

# ------------------------------------------------ 3. eng ko'p klipli spikerning audiosini yig'ish
best_spk, best_rows = top[0]
best_rows.sort(key=lambda r: order_key(r["id"]))
want = {r["id"] for r in best_rows}

audio = {}
for sh in SHARDS:
    p = hf_hub_download(REPO, sh, repo_type="dataset")
    t = pq.read_table(p)
    ids = t.column("id").to_pylist()
    auds = t.column("audio")
    for i, uid in enumerate(ids):
        if uid in want:
            a = auds[i].as_py()
            d, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            if d.ndim > 1:
                d = d.mean(axis=1)
            if sr != SR:
                import librosa

                d = librosa.resample(d, orig_sr=sr, target_sr=SR)
            audio[uid] = d
print(f"[3] spiker {best_spk}: {len(audio)} klip audio o'qildi")

# ------------------------------------------------ 4. ulash (TARGET_MIN gacha)
gap = np.zeros(int(GAP * SR), dtype=np.float32)
pieces, truth, cursor, used = [], [], 0.0, []
for r in best_rows:
    if cursor >= TARGET_MIN * 60:
        break
    d = audio.get(r["id"])
    if d is None or len(d) < SR * 0.5:
        continue
    start = cursor
    pieces.append(d)
    cursor += len(d) / SR
    truth.append({"start": round(start, 3), "end": round(cursor, 3),
                  "text": r["sentence"], "id": r["id"]})
    used.append(r)
    pieces.append(gap)
    cursor += GAP

long_wav = np.concatenate(pieces)
sf.write(os.path.join(OUT, "long.wav"), long_wav, SR, subtype="PCM_16")
text = " ".join(t["text"] for t in truth)
io.open(os.path.join(OUT, "text.txt"), "w", encoding="utf-8").write(text)
json.dump(truth, io.open(os.path.join(OUT, "truth.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

dur_min = len(long_wav) / SR / 60
print(f"[4] long.wav: {dur_min:.2f} daqiqa | {len(truth)} klip | {len(text.split())} so'z")

# ------------------------------------------------ 5. foydalanuvchi uchun namuna
md = [f"# USC sinov namunasi — spiker `{best_spk}`", "",
      f"- Manba: `{REPO}` (CC-BY-4.0), test bo'limi",
      f"- Davomiylik: **{dur_min:.2f} daqiqa**, {len(truth)} klip, {len(text.split())} so'z",
      f"- Kliplar orasida {GAP} s jimlik qo'shilgan (tabiiy gap pauzasi)",
      "- Matn USC'da allaqachon normalizatsiyalangan: kichik harf, punktuatsiyasiz",
      "", "## Birinchi 12 gap (audio bilan mosligini tekshiring)", "",
      "| # | vaqt | gap |", "|---|---|---|"]
for i, t in enumerate(truth[:12], 1):
    md.append(f"| {i} | {t['start']:.1f}–{t['end']:.1f} s | {t['text']} |")
md += ["", "## Oxirgi 5 gap", "", "| # | vaqt | gap |", "|---|---|---|"]
for i, t in enumerate(truth[-5:], len(truth) - 4):
    md.append(f"| {i} | {t['start']:.1f}–{t['end']:.1f} s | {t['text']} |")
io.open(os.path.join(OUT, "namuna.md"), "w", encoding="utf-8").write("\n".join(md))
print(f"[5] namuna.md yozildi -> {OUT}")

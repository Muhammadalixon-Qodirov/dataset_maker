"""Kesilgan chetlarda NUTQ bor-yo'qligini tekshirish.

Agar produced.start > truth.start bo'lsa, oradagi qismda nutq bo'lsa —
biz so'zni kesib tashlayapmiz (haqiqiy xato). Jimlik bo'lsa — muammo yo'q.
"""

import csv
import io
import json
import os

import numpy as np
import soundfile as sf

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleurs_test")
truth = json.load(io.open(os.path.join(B, "truth.json"), encoding="utf-8"))
wav, sr = sf.read(os.path.join(B, "long.wav"), dtype="float32")

rows = list(csv.reader(io.open(os.path.join(B, "out", "segments.csv"), encoding="utf-8"),
                       delimiter="|"))[1:]
prod = [{"start": float(r[1]), "end": float(r[2])} for r in rows if len(r) >= 3]

GLOBAL_TH = float(np.abs(wav).max()) * 0.02


def has_speech(t0: float, t1: float) -> tuple[bool, float]:
    """Oraliqda nutq bormi + eng baland sample."""
    a, b = int(t0 * sr), int(t1 * sr)
    if b <= a:
        return False, 0.0
    seg = np.abs(wav[a:b])
    return bool((seg > GLOBAL_TH).mean() > 0.05), float(seg.max())


def overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


cut_start, cut_end, ok_start, ok_end = 0, 0, 0, 0
examples = []
used = set()
for t in truth:
    best, bo = None, 0.0
    for i, p in enumerate(prod):
        if i in used:
            continue
        o = overlap(t, p)
        if o > bo:
            best, bo = i, o
    if best is None or bo < 0.5:
        continue
    used.add(best)
    p = prod[best]

    if p["start"] > t["start"] + 0.05:
        sp, amp = has_speech(t["start"], p["start"])
        (cut_start := cut_start + 1) if sp else (ok_start := ok_start + 1)
        if sp and len(examples) < 6:
            examples.append(("START", t["start"], p["start"],
                             (p["start"] - t["start"]) * 1000, amp))
    if p["end"] < t["end"] - 0.05:
        sp, amp = has_speech(p["end"], t["end"])
        (cut_end := cut_end + 1) if sp else (ok_end := ok_end + 1)
        if sp and len(examples) < 6:
            examples.append(("END", p["end"], t["end"],
                             (t["end"] - p["end"]) * 1000, amp))

print(f"juftlashtirilgan: {len(used)}")
print()
print("START chetida (produced kech boshlandi):")
print(f"  kesilgan joyda NUTQ bor  : {cut_start}   <-- MUAMMO")
print(f"  kesilgan joyda jimlik    : {ok_start}   <-- muammo emas")
print()
print("END chetida (produced erta tugadi):")
print(f"  kesilgan joyda NUTQ bor  : {cut_end}   <-- MUAMMO")
print(f"  kesilgan joyda jimlik    : {ok_end}   <-- muammo emas")
print()
if examples:
    print("namunalar (nutq kesilgan joylar):")
    for kind, a, b, ms, amp in examples:
        print(f"  {kind:5} {a:7.2f} -> {b:7.2f}  ({ms:6.0f}ms, amp {amp:.3f})")
else:
    print("nutq kesilgan joy TOPILMADI — chetlar jim")

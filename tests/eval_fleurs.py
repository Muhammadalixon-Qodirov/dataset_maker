"""Tizim natijasini FLEURS haqiqiy chegaralari bilan solishtirish.

O'lchanadi:
  - recall: 64 haqiqiy segmentdan nechtasi topildi
  - chegara xatosi: start/end millisekundda (belgili — pad siljishi ko'rinadi)
  - matn aniqligi: ishlab chiqarilgan matn vs haqiqiy matn CER
"""

import csv
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quality import cer
from src.uz_text import romanize_for_align

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleurs_test")
truth = json.load(io.open(os.path.join(BASE, "truth.json"), encoding="utf-8"))
OUT = os.path.join(BASE, "out")


def read_csv(name, has_header=True):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return []
    rows = list(csv.reader(io.open(p, encoding="utf-8"), delimiter="|"))
    return rows[1:] if has_header and rows else rows


meta = {r[0]: r[1] for r in read_csv("metadata.csv", has_header=False) if len(r) >= 2}
segs = [r for r in read_csv("segments.csv") if len(r) >= 3]
rej = read_csv("rejected.csv")

print(f"haqiqiy segment : {len(truth)}")
print(f"ishlab chiqarilgan: {len(segs)}")
print(f"rad etilgan      : {len(rej)}")

if not segs:
    print("\nsegments.csv bo'sh — eski kod bilan ishlatilgan bo'lishi mumkin")
    sys.exit(0)

produced = []
for r in segs:
    produced.append({
        "name": r[0], "start": float(r[1]), "end": float(r[2]),
        "score": float(r[3]) if r[3] else None,
        "cer": float(r[4]) if len(r) > 4 and r[4] else None,
        "text": meta.get(r[0], ""),
    })

# ------------------------------------------- vaqt bo'yicha juftlash (IoU)
def overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


matched, start_err, end_err, cers = [], [], [], []
used = set()
for t in truth:
    best, best_ov = None, 0.0
    for i, p in enumerate(produced):
        if i in used:
            continue
        ov = overlap(t, p)
        if ov > best_ov:
            best, best_ov = i, ov
    if best is None or best_ov < 0.5:
        continue
    used.add(best)
    p = produced[best]
    matched.append((t, p))
    start_err.append((p["start"] - t["start"]) * 1000)
    end_err.append((p["end"] - t["end"]) * 1000)
    cers.append(cer(romanize_for_align(t["text"]), romanize_for_align(p["text"])))

print(f"\njuftlashtirildi  : {len(matched)}/{len(truth)}  "
      f"(recall {len(matched)/len(truth)*100:.1f}%)")

if not matched:
    sys.exit(0)


def stats(vals, unit=""):
    s = sorted(vals)
    n = len(s)
    return (f"median {s[n//2]:+.0f}{unit} | o'rtacha {sum(s)/n:+.0f}{unit} | "
            f"min {s[0]:+.0f}{unit} | max {s[-1]:+.0f}{unit}")


print(f"\nSTART xatosi : {stats(start_err, 'ms')}")
print(f"END   xatosi : {stats(end_err, 'ms')}")
print("  (manfiy = oldin boshlandi · musbat = keyin tugadi)")
print("  PAD=0.25s tufayli start ~-250ms, end ~+250ms kutiladi — bu DIZAYN")

abs_all = [abs(x) for x in start_err + end_err]
for th in (100, 250, 500, 1000):
    k = sum(1 for x in abs_all if x <= th)
    print(f"  |xato| <= {th:>4}ms : {k}/{len(abs_all)} ({k/len(abs_all)*100:.1f}%)")

# pad ta'sirini olib tashlab, sof siljishni ko'rish
corr = [abs(x + 250) for x in start_err] + [abs(x - 250) for x in end_err]
k100 = sum(1 for x in corr if x <= 100)
print(f"\nPAD tuzatilgandan keyin |xato| <= 100ms : "
      f"{k100}/{len(corr)} ({k100/len(corr)*100:.1f}%)")

print(f"\nMATN CER : median {sorted(cers)[len(cers)//2]*100:.2f}% | "
      f"o'rtacha {sum(cers)/len(cers)*100:.2f}% | max {max(cers)*100:.2f}%")
perfect = sum(1 for c in cers if c == 0.0)
print(f"  mukammal mos (CER=0) : {perfect}/{len(cers)} ({perfect/len(cers)*100:.1f}%)")

scores = [p["score"] for _, p in matched if p["score"] is not None]
if scores:
    s = sorted(scores)
    print(f"\nmin_score (qabul qilinganlar) : median {s[len(s)//2]:.3f} | "
          f"min {s[0]:.3f} | max {s[-1]:.3f}")

print("\n--- eng yomon 5 chegara ---")
worst = sorted(zip(matched, start_err, end_err),
               key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)[:5]
for (t, p), se, ee in worst:
    print(f"  start {se:+7.0f}ms  end {ee:+7.0f}ms  | {t['text'][:55]}")

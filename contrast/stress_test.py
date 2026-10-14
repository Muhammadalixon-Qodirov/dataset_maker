"""Stress test: run the detector on every file of the contrastive dataset.

    python -m contrast.stress_test       # -> contrast_data/stress_report.json + .md

The detector never sees the labels. Each participant file is aligned from scratch and
compared with its baseline, then predictions are scored against the injected ground truth:

    detection   a GT flaw counts as found if a predicted region of the SAME kind overlaps it
                (temporal IoU >= 0.2, or for point-like flaws: overlap at all)
    precision   predicted regions that overlap a GT flaw of the same kind / all predictions
    boundary    |start error| and |end error| in seconds for found flaws
    gradient    rubric score per severity level must fall as severity rises
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

from . import compare as C
from . import rubric as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "contrast_data", "dataset")
POINT = {"long_pause", "stutter"}


def iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match(gt: dict, preds: list[dict]) -> dict | None:
    best, best_iou = None, -1.0
    for p in preds:
        if p["kind"] != gt["type"]:
            continue
        v = iou(gt["start"], gt["end"], p["start"], p["end"])
        ok = v > 0 if gt["type"] in POINT else v >= 0.2
        if ok and v > best_iou:
            best, best_iou = p, v
    return best


def main() -> None:
    rows = list(csv.DictReader(open(os.path.join(DS, "metadata.csv"), encoding="utf-8")))
    bases: dict[str, C.Aligned] = {}
    found = defaultdict(lambda: [0, 0])  # (type, sev) -> [found, total]
    bnd = defaultdict(list)
    n_pred = n_pred_ok = 0
    scores = defaultdict(list)  # (type, sev) -> overall
    per_file = []
    for r in rows:
        meta = json.load(open(os.path.join(DS, r["file"].replace(".flac", ".json")), encoding="utf-8"))
        sp = r["speech"]
        if sp not in bases:
            bw, _ = sf.read(os.path.join(DS, sp, "baseline.flac"), dtype="float32")
            bases[sp] = C.align(bw, meta["transcript"])
        pw, _ = sf.read(os.path.join(DS, r["file"]), dtype="float32")
        res = C.compare(bases[sp], C.align(pw, meta["transcript"]))
        preds = C.as_dicts(res["regions"])
        rub = R.score(preds)
        gts = meta["flaws"]
        for g in gts:
            m = match(g, preds)
            key = (g["type"], g["severity"])
            found[key][1] += 1
            if m:
                found[key][0] += 1
                bnd[g["type"]].append((abs(m["start"] - g["start"]), abs(m["end"] - g["end"])))
        for p in preds:
            n_pred += 1
            n_pred_ok += any(p["kind"] == g["type"] and iou(g["start"], g["end"], p["start"], p["end"]) > 0
                             for g in gts)
        if r["kind"] == "single":
            scores[(gts[0]["type"], int(r["severity"]))].append(rub["overall"])
        else:
            scores[(r["kind"] if r["kind"] == "baseline" else os.path.basename(r["file"])[:-5], 0)].append(rub["overall"])
        per_file.append({"file": r["file"], "gt": [g["type"] for g in gts], "pred": [p["kind"] for p in preds],
                         "overall": rub["overall"], "criteria": rub["criteria"]})
        print(f"{r['file']:<38} gt={','.join(g['type'] for g in gts) or '-':<38} "
              f"pred={','.join(p['kind'] for p in preds) or '-':<40} score={rub['overall']}")

    types = sorted({t for t, _ in found})
    det = {t: {s: found[(t, s)] for s in (1, 2, 3) if (t, s) in found} for t in types}
    total_f = sum(v[0] for v in found.values())
    total_n = sum(v[1] for v in found.values())
    sev = {s: [sum(found[(t, s)][0] for t in types if (t, s) in found),
               sum(found[(t, s)][1] for t in types if (t, s) in found)] for s in (1, 2, 3)}
    grad = {t: [round(float(np.mean(scores[(t, s)])), 2) for s in (1, 2, 3)] for t in types if (t, 1) in scores}
    mono = sum(1 for v in grad.values() if v[0] >= v[1] >= v[2])
    report = {
        "files": len(rows),
        "recall_overall": round(total_f / total_n, 3),
        "recall_by_severity": {s: round(v[0] / v[1], 3) if v[1] else None for s, v in sev.items()},
        "precision": round(n_pred_ok / n_pred, 3) if n_pred else None,
        "false_alarms_on_baselines": sum(len(f["pred"]) for f in per_file if f["file"].endswith("baseline.flac")),
        "detection": {t: {s: f"{v[0]}/{v[1]}" for s, v in d.items()} for t, d in det.items()},
        "boundary_error_s": {t: {"start_median": round(float(np.median([a for a, _ in v])), 3),
                                 "end_median": round(float(np.median([b for _, b in v])), 3)}
                             for t, v in bnd.items() if v},
        "rubric_by_severity": grad,
        "rubric_monotonic_types": f"{mono}/{len(grad)}",
        "rubric_composites": {k[0]: round(float(np.mean(v)), 2) for k, v in scores.items() if k[1] == 0},
        "per_file": per_file,
    }
    out = os.path.join(ROOT, "contrast_data", "stress_report.json")
    json.dump(report, open(out, "w"), indent=1)

    md = ["# Stress test report", "",
          f"{len(rows)} files · recall **{report['recall_overall']:.0%}** "
          f"(severity 1: {report['recall_by_severity'][1]:.0%}, 2: {report['recall_by_severity'][2]:.0%}, "
          f"3: {report['recall_by_severity'][3]:.0%}) · precision **{report['precision']:.0%}** · "
          f"false alarms on clean baselines: **{report['false_alarms_on_baselines']}**", "",
          "| Flaw | sev 1 | sev 2 | sev 3 | start err (median) | end err (median) | score s1 → s2 → s3 |",
          "|---|---|---|---|---|---|---|"]
    for t in types:
        b = report["boundary_error_s"].get(t, {})
        g = grad.get(t, ["", "", ""])
        md.append(f"| {t} | {det[t].get(1, ['-'])[0]}/{det[t].get(1, ['-', '-'])[1]} | "
                  f"{det[t].get(2, ['-'])[0]}/{det[t].get(2, ['-', '-'])[1]} | "
                  f"{det[t].get(3, ['-'])[0]}/{det[t].get(3, ['-', '-'])[1]} | "
                  f"{b.get('start_median', '-')} s | {b.get('end_median', '-')} s | {g[0]} → {g[1]} → {g[2]} |")
    md += ["", f"Rubric falls monotonically with severity for {report['rubric_monotonic_types']} flaw types.",
           f"Composite files: {report['rubric_composites']}"]
    open(out.replace(".json", ".md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Reproducible rubric: 0-10 per criterion, deterministic (no sampling, no LLM).

For each criterion, every detected region adds a penalty

    penalty_r = magnitude_r x (0.5 + 0.5 x min(1, duration_r / 5 s))

so a large deviation counts more, and a long one more than a short one. The criterion
score is 10 x max(0, 1 - sum of penalties). Overall = mean of the five criteria.
Same audio + same transcript -> same score, every run.
"""

from __future__ import annotations

CRITERIA = ["Pace", "Pausing", "Expressiveness", "Volume & clarity", "Fluency"]
WHAT = {
    "Pace": "syllable rate stays close to the reference delivery",
    "Pausing": "no unplanned silences",
    "Expressiveness": "pitch keeps moving to carry emphasis",
    "Volume & clarity": "loud and bright enough to be understood",
    "Fluency": "no repeated onsets or sounds outside the text",
}


def score(regions: list[dict]) -> dict:
    pen = {c: 0.0 for c in CRITERIA}
    for r in regions:
        dur = max(0.0, r["end"] - r["start"])
        pen[r["criterion"]] += r["magnitude"] * (0.5 + 0.5 * min(1.0, dur / 5.0))
    per = {c: round(10 * max(0.0, 1 - p), 1) for c, p in pen.items()}
    overall = round(sum(per.values()) / len(per), 1)
    return {"criteria": per, "overall": overall}

"""Reproducible rubric: 0-10 per criterion, deterministic (no sampling, no LLM).

Two sources of penalty, both read from compare()'s result:

  local regions   penalty = magnitude x (0.5 + 0.5 x min(1, duration / 5 s))
                  a large deviation counts more, a long one more than a short one
  global profile  overall tempo outside +/-15%, pauses > 1.4x the reference,
                  pitch range < 80% of the reference: a whole-speech habit

Criterion = 10 x max(0, 1 - sum of its penalties); overall = mean of the five.
Every penalty is returned as an item, so the coach can rank what to practise first.
"""

from __future__ import annotations

import math

CRITERIA = ["Pace", "Pausing", "Expressiveness", "Volume & clarity", "Fluency"]
WHAT = {
    "Pace": "syllable rate stays close to the reference delivery",
    "Pausing": "no unplanned silences",
    "Expressiveness": "pitch keeps moving to carry emphasis",
    "Volume & clarity": "loud, bright and well-articulated",
    "Fluency": "no repeated onsets or sounds outside the text",
}


def profile_items(profile: dict | None) -> list[dict]:
    if not profile:
        return []
    items = []
    t = profile["tempo_ratio"]
    if abs(math.log(t)) > math.log(1.15):
        items.append({"criterion": "Pace", "source": "profile:tempo",
                      "penalty": 0.5 * min(1.0, (abs(math.log(t)) - math.log(1.15)) / math.log(2) * 2)})
    p = profile["pause_ratio"]
    if p > 1.4:
        items.append({"criterion": "Pausing", "source": "profile:pauses",
                      "penalty": 0.4 * min(1.0, (math.log(p) - math.log(1.4)) / math.log(2) * 2)})
    r = profile["pitch_range_ratio"]
    if r < 0.8:
        items.append({"criterion": "Expressiveness", "source": "profile:pitch",
                      "penalty": 0.5 * min(1.0, (0.8 - r) / 0.4)})
    return items


def score(regions: list[dict], profile: dict | None = None) -> dict:
    items = []
    for k, r in enumerate(regions):
        dur = max(0.0, r["end"] - r["start"])
        items.append({"criterion": r["criterion"], "source": f"region:{k}",
                      "penalty": r["magnitude"] * (0.5 + 0.5 * min(1.0, dur / 5.0))})
    items += profile_items(profile)
    pen = {c: 0.0 for c in CRITERIA}
    for it in items:
        pen[it["criterion"]] += it["penalty"]
    per = {c: round(10 * max(0.0, 1 - p), 1) for c, p in pen.items()}
    return {"criteria": per, "overall": round(sum(per.values()) / len(per), 1),
            "items": sorted(items, key=lambda x: -x["penalty"])}

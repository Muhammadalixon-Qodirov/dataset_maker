"""Coaching: turns compare()'s numbers into actions, and ranks them by rubric penalty.

    tip(region)          one concrete instruction per flaw, with the target number
    profile_findings()   whole-speech habits from the profile (tempo, pauses, pitch range)
    practice_plan()      the top items by rubric penalty -> "practise these first"
    snippets()           reference vs participant audio for one region (A/B listening)

Nothing here measures anything: it only reads compare() and rubric.score() output, so the
tips, the plan, the A/B clips and the PDF report always agree with the dashboard.
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

SR = 16000


def tip(r: dict) -> str:
    d, k = r.get("data", {}), r["kind"]
    w = r["words"]
    if k == "rushed":
        return (f"Slow down on “{w}”: aim for about {d['target']:.1f} syllables/s (you were at {d['rp']:.1f}). "
                "Take a breath before the phrase and land each stressed syllable.")
    if k == "dragging":
        return (f"Pick up the pace on “{w}”: aim for about {d['target']:.1f} syllables/s (you were at {d['rp']:.1f}). "
                "Keep vowels short and carry the phrase on one breath.")
    if k == "long_pause":
        return (f"Close the gap at “{w}”: about {d['target']:.2f} s is enough here (you paused {d['pp']:.2f} s). "
                "Mark the next phrase in your script so the pause is a choice, not a search.")
    if k == "monotone":
        need = max(1.0, d["b"] - d["p"])
        return (f"Give “{w}” a melody: lift your pitch on “{d['stress_word']}” by about {need:.0f} semitone"
                f"{'s' if need >= 1.5 else ''} and let it fall at the end of the phrase "
                f"(the reference moves {d['b']:.1f} st here, you {d['p']:.1f}).")
    if k == "mumble":
        return (f"Project on “{w}”: about {abs(d['db']):.0f} dB louder, mouth open on the consonants, "
                "voice aimed at the back of the room.")
    if k == "stutter":
        return (f"Restart cleanly on “{w}”: if the word catches, stop for a beat and say it whole. "
                "Practise the phrase slowly three times, then at full speed.")
    if k == "unclear":
        return (f"Articulate “{w}” fully: exaggerate its consonants three times slowly, then say it at "
                "normal speed. Listen to the reference clip and copy its shape.")
    return ""


def profile_findings(p: dict) -> list[dict]:
    """Whole-speech habits. Each finding carries the rubric source id it explains."""
    out = []
    t = p["tempo_ratio"]
    if t < 0.87:
        out.append({"source": "profile:tempo", "label": "Overall slow tempo",
                    "text": f"Overall you speak {(1 - t) * 100:.0f}% slower than the reference.",
                    "tip": "Run the whole speech once at the reference pace with a timer: "
                           f"the reference takes about {p['duration_s']['reference']:.0f} s, "
                           f"you took {p['duration_s']['participant']:.0f} s."})
    elif t > 1.15:
        out.append({"source": "profile:tempo", "label": "Overall fast tempo",
                    "text": f"Overall you speak {(t - 1) * 100:.0f}% faster than the reference.",
                    "tip": "Slow the whole delivery down: pause at every full stop and finish each sentence before the next."})
    pr = p["pause_ratio"]
    if pr > 1.4:
        out.append({"source": "profile:pauses", "label": "Long pauses throughout",
                    "text": f"Your pauses between phrases are {pr:.1f}× the reference's.",
                    "tip": "Rehearse with the script in hand until each phrase starts as soon as the last one ends."})
    elif pr < 0.6:
        out.append({"source": "profile:pauses", "label": "Too few pauses",
                    "text": f"Your pauses are only {pr:.1f}× the reference's.",
                    "tip": "Let full stops breathe: half a second of silence lets the audience keep up."})
    g = p["pitch_range_ratio"]
    if g < 0.8:
        out.append({"source": "profile:pitch", "label": "Narrow pitch range",
                    "text": f"Your pitch moves {(1 - g) * 100:.0f}% less than the reference's across the speech.",
                    "tip": "Pick one key word per sentence and make it clearly higher than the rest."})
    return out


def practice_plan(regions: list[dict], findings: list[dict], rubric: dict, n: int = 3) -> list[dict]:
    """The n rubric items that cost the most points, each with its action."""
    by_src = {f"region:{k}": r for k, r in enumerate(regions)}
    fin = {f["source"]: f for f in findings}
    plan = []
    for it in rubric["items"]:
        src = it["source"]
        if src in by_src:
            r = by_src[src]
            plan.append({"criterion": it["criterion"], "points": round(10 * it["penalty"] / 5, 2),
                         "what": f"{r['label']} at {r['start']:.1f}–{r['end']:.1f} s", "do": r["tip"], "region": src})
        elif src in fin:
            f = fin[src]
            plan.append({"criterion": it["criterion"], "points": round(10 * it["penalty"] / 5, 2),
                         "what": f["label"], "do": f["tip"], "region": None})
        if len(plan) == n:
            break
    return plan


def enrich(regions: list[dict]) -> list[dict]:
    for r in regions:
        r["tip"] = tip(r)
    return regions


def snippets(res: dict, r: dict, out_dir: str, pad: float = 0.35) -> tuple[str, str]:
    """A/B clips of the same words: reference first, then the participant."""
    wb, wp = res["base_words"], res["part_words"]
    i, j = r["i"], r["j"]
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, wav, ws in (("reference", res["base_wav"], wb), ("you", res["part_wav"], wp)):
        a = max(0, int((ws[i].start - pad) * SR))
        b = min(wav.size, int((ws[j].end + pad) * SR))
        clip = wav[a:b].astype(np.float32)
        fade = min(len(clip) // 4, int(0.02 * SR))
        if fade > 0:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        p = os.path.join(out_dir, f"{name}_{i}_{j}.wav")
        sf.write(p, clip, SR)
        paths.append(p)
    return paths[0], paths[1]

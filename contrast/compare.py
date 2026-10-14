"""Baseline vs participant: word-level comparison -> grounded flaw regions + causal text.

Both recordings are force-aligned to the SAME transcript (MMS_FA), so word k of one
corresponds to word k of the other. Every feature is compared word by word, and a
deviation is expressed two ways:

    ratio / delta     in natural units (syllables/s, seconds, semitones, dB)
    z                 deviation divided by the BASELINE speaker's own natural variability
                      of that feature across the speech (how unusual it is for good delivery)

A word is flagged when both the effect size and z pass a threshold; consecutive flagged
words of the same kind merge into one region with exact start/end timestamps.
Extra sounds between words (stutters, fillers) are caught by the aligner's wildcard token.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from src import align as A

from . import features as F
from . import text_en as T

SR = 16000

# thresholds — natural-unit effect size AND z must both pass
RATE_RATIO = 1.18  # >18% faster/slower than baseline over a 3-word window
PAUSE_DELTA = 0.45  # s of extra silence
PITCH_SD_RATIO = 0.62  # pitch movement shrinks below 62% of baseline
ENERGY_DB = -5.0  # quieter than baseline, re each speaker's own median
CENTROID_RATIO = 0.78  # spectral centroid drop = muffled articulation
EXTRA_SOUND = 0.15  # s of speech-like sound the transcript does not contain
Z_MIN = 1.5

KIND_LABEL = {
    "rushed": "Rushed pace", "dragging": "Dragging pace", "long_pause": "Unplanned long pause",
    "monotone": "Monotone delivery", "mumble": "Quiet / muffled", "stutter": "Disfluency (repeat/stutter)",
}
CRITERION = {
    "rushed": "Pace", "dragging": "Pace", "long_pause": "Pausing", "monotone": "Expressiveness",
    "mumble": "Volume & clarity", "stutter": "Fluency",
}


@dataclass
class Aligned:
    words: list  # WordSpan, transcript words only
    extras: list  # WordSpan of wildcard (*) spans = sound not in the transcript
    display: list[str]
    wav: np.ndarray


@dataclass
class Region:
    kind: str
    start: float
    end: float
    words: str
    magnitude: float  # 0..1 normalized deviation (drives the rubric)
    z: float
    explanation: str


_ALIGNER: A.ForcedAligner | None = None


def aligner() -> A.ForcedAligner:
    global _ALIGNER
    if _ALIGNER is None:
        _ALIGNER = A.ForcedAligner(device="cuda" if torch.cuda.is_available() else "cpu")
    return _ALIGNER


def align(wav: np.ndarray, transcript: str) -> Aligned:
    pairs = T.split_aligned(transcript)
    words = [al for _, al in pairs]
    al = aligner()
    em = al.emissions(torch.from_numpy(wav.astype(np.float32)), chunk_sec=20.0)
    spans = al.align(em, A.add_stars(words, "between"))
    real = [s for s in spans if s.word != A.STAR]
    stars = [s for s in spans if s.word == A.STAR]
    if len(real) != len(words):
        raise ValueError(f"aligned {len(real)}/{len(words)} words")
    display = []
    for (d, _), s in zip(pairs, real):
        display.append(d or s.word)
    for d, s in zip(display, real):
        s.word = d  # keep punctuation-free display for features, syllables use letters only
    return Aligned(real, stars, display, wav)


def _robust_sd(x: np.ndarray, floor: float) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return floor
    mad = float(np.median(np.abs(x - np.median(x)))) * 1.4826
    return max(mad, floor)


def _regions(flags: list[tuple[int, float, float]], wf: list[F.WordFeat], kind: str) -> list[tuple]:
    """Merge consecutive flagged word indices -> [(i, j, mean_mag, max_z)]."""
    out, cur = [], []
    for i, mag, z in sorted(flags):
        if cur and i != cur[-1][0] + 1:
            out.append(cur)
            cur = []
        cur.append((i, mag, z))
    if cur:
        out.append(cur)
    return [(c[0][0], c[-1][0], float(np.mean([m for _, m, _ in c])), float(max((z for _, _, z in c), key=abs)))
            for c in out]


def _speaking(ws: list, i: int, j: int) -> float:
    """Time from word i to word j minus silences > 0.2 s (articulation time)."""
    return ws[j].end - ws[i].start - sum(max(0.0, ws[k].start - ws[k - 1].end - 0.2) for k in range(i + 1, j + 1))


def compare(base: Aligned, part: Aligned) -> dict:
    fb, fp = F.frames(base.wav), F.frames(part.wav)
    wb, wp = F.word_features(fb, base.words), F.word_features(fp, part.words)
    n = len(wb)
    regions: list[Region] = []

    # ---- pace: log-ratio of local syllable rate
    lr = np.log(np.array([p.rate / b.rate for b, p in zip(wb, wp)]))
    # scale = typical word-to-word difference between the two readings (most words are
    # delivered the same, so this is the measurement noise, not the speaker's variability)
    sd = _robust_sd(lr - np.nanmedian(lr), 0.05)
    for kind, sign in (("rushed", 1), ("dragging", -1)):
        flags = [(i, min(1.0, abs(lr[i]) / np.log(2)), lr[i] / sd) for i in range(n)
                 if sign * lr[i] > np.log(RATE_RATIO) and sign * lr[i] / sd > Z_MIN]
        for i, j, mag, z in _regions(flags, wp, kind):
            rb = sum(w.syl for w in wb[i:j + 1]) / max(1e-3, _speaking(wb, i, j))
            rp = sum(w.syl for w in wp[i:j + 1]) / max(1e-3, _speaking(wp, i, j))
            if sign * np.log(rp / rb) < np.log(RATE_RATIO):
                continue  # word-level noise that does not hold over the whole region
            pct = (rp / rb - 1) * 100
            regions.append(Region(kind, wp[i].start, wp[j].end, " ".join(w.word for w in wp[i:j + 1]), mag, z,
                f"Speech rate {rp:.1f} syl/s vs baseline {rb:.1f} syl/s ({pct:+.0f}%, z={z:+.1f}). "
                + ("Words are compressed; listeners lose the content." if kind == "rushed"
                   else "Words are drawn out; the line loses momentum.")))

    # ---- pausing: extra silence before a word
    floor_p = float(np.nanpercentile(fp.energy_db, 5))
    pb = np.array([w.pause_before for w in wb])
    sdp = _robust_sd(pb, 0.12)
    for i in range(1, n):
        d = wp[i].pause_before - wb[i].pause_before
        z = d / sdp
        gap = fp.energy_db[F._sl(fp, wp[i - 1].end, wp[i].start)]
        # a "gap" that still holds speech energy is mis-aligned quiet speech, not a pause
        # quiet = near this recording's own noise floor (old recordings have loud room tone)
        quiet = gap.size == 0 or float(np.mean(gap > floor_p + 10)) < 0.3
        if d > PAUSE_DELTA and z > Z_MIN and quiet:
            regions.append(Region("long_pause", wp[i - 1].end, wp[i].start, f"{wp[i-1].word} … {wp[i].word}",
                min(1.0, d / 2.5), z,
                f"Silence of {wp[i].pause_before:.2f} s vs {wb[i].pause_before:.2f} s in the baseline "
                f"(+{d:.2f} s, z={z:+.1f}) between \"{wp[i-1].word}\" and \"{wp[i].word}\". "
                "A pause this long reads as a lost line, not emphasis."))

    # ---- expressiveness: pitch movement over a 5-word window (semitones)
    sb = np.array([w.pitch_sd for w in wb])
    spv = np.array([w.pitch_sd for w in wp])
    ratio = spv / sb
    sds = _robust_sd((spv - sb) - np.nanmedian(spv - sb), 0.2)
    flags = [(i, min(1.0, 1 - ratio[i]), (spv[i] - sb[i]) / sds) for i in range(n)
             if np.isfinite(ratio[i]) and sb[i] > 1.0 and ratio[i] < PITCH_SD_RATIO
             and (sb[i] - spv[i]) / sds > Z_MIN]
    for i, j, mag, z in _regions(flags, wp, "monotone"):
        b_, p_ = float(np.nanmean(sb[i:j + 1])), float(np.nanmean(spv[i:j + 1]))
        regions.append(Region("monotone", wp[i].start, wp[j].end, " ".join(w.word for w in wp[i:j + 1]), mag, z,
            f"Pitch movement {p_:.1f} st vs baseline {b_:.1f} st ({(p_ / b_ - 1) * 100:+.0f}%, z={z:+.1f}). "
            "Intonation is flat, so key words get no emphasis."))

    # ---- volume & clarity: energy re own median, spectral centroid
    eb, ep = np.array([w.energy for w in wb]), np.array([w.energy for w in wp])
    cb, cp = np.array([w.centroid for w in wb]), np.array([w.centroid for w in wp])
    de, cr = ep - eb, cp / cb
    sde = _robust_sd(de - np.nanmedian(de), 1.0)
    # energy drop is required; a centroid drop alone is too often a word-boundary artefact
    flags = [(i, min(1.0, max(-de[i] / 20, (1 - cr[i]) / 0.6 if cr[i] < CENTROID_RATIO else 0)), de[i] / sde)
             for i in range(n) if np.isfinite(de[i]) and de[i] < ENERGY_DB and de[i] / sde < -Z_MIN]
    for i, j, mag, z in _regions(flags, wp, "mumble"):
        d_ = float(np.nanmean(de[i:j + 1]))
        c_ = float(np.nanmean(cr[i:j + 1]))
        regions.append(Region("mumble", wp[i].start, wp[j].end, " ".join(w.word for w in wp[i:j + 1]), mag, z,
            f"Energy {d_:+.1f} dB vs baseline (z={z:+.1f}); spectral centroid {(c_ - 1) * 100:+.0f}% "
            f"({float(np.nanmean(cp[i:j+1])):.0f} vs {float(np.nanmean(cb[i:j+1])):.0f} Hz). "
            "Quieter and duller: consonants blur and words are hard to hear."))

    # ---- fluency: sound between words that the transcript doesn't contain.
    # Good speakers also breathe and co-articulate, so the wildcard catches some sound
    # in the baseline too: only the EXCESS over the baseline at the same word counts.
    def extra_before(al: Aligned, fr: F.Frames) -> dict[int, tuple[float, float, float]]:
        out = {}
        for s in al.extras:
            k = next((i for i, w in enumerate(al.words) if w.start >= s.end - 0.02), None)
            if k is None or k == 0:
                continue  # leading/trailing room noise
            loud = fr.energy_db[F._sl(fr, s.start, s.end)]
            vf = float(np.mean(loud > -12)) if loud.size else 0.0
            out[k] = (s.end - s.start, vf, s.start)
        return out

    xb, xp = extra_before(base, fb), extra_before(part, fp)
    cand = []
    for k, (dur, vf, st) in xp.items():
        bd, bv, _ = xb.get(k, (0.0, 0.0, 0.0))
        excess = dur * vf - bd * bv
        if excess >= EXTRA_SOUND and vf >= 0.35 and dur * vf >= 1.5 * bd * bv:
            cand.append((k, dur, excess, st))
    # extra sound before SEVERAL consecutive words = the phrase is drawn out (dragging),
    # a single word with extra sound in front of it = a repeated onset / filler (stutter)
    groups, cur = [], []
    for c in sorted(cand):
        if cur and c[0] > cur[-1][0] + 1:
            groups.append(cur)
            cur = []
        cur.append(c)
    if cur:
        groups.append(cur)
    for g in groups:
        # repeated onsets are short (3 repeats of ~0.3 s); a longer block of extra sound
        # in front of one word is a drawn-out phrase the aligner could not split
        if len(g) >= 2 or g[0][1] > 1.2:
            i, j = max(0, g[0][0] - 3 if len(g) == 1 else g[0][0] - 1), min(n - 1, g[-1][0] + (1 if len(g) == 1 else 0))
            rb = sum(w.syl for w in wb[i:j + 1]) / max(1e-3, wb[j].end - wb[i].start)
            rp = sum(w.syl for w in wp[i:j + 1]) / max(1e-3, wp[j].end - wp[i].start)
            if rp < rb / RATE_RATIO:
                regions.append(Region("dragging", wp[i].start, wp[j].end, " ".join(w.word for w in wp[i:j + 1]),
                    min(1.0, abs(np.log(rp / rb)) / np.log(2)), -sum(x[2] for x in g) / 0.1,
                    f"Phrase takes {wp[j].end - wp[i].start:.2f} s vs {wb[j].end - wb[i].start:.2f} s in the baseline "
                    f"({rp:.1f} vs {rb:.1f} syl/s, {(rp / rb - 1) * 100:+.0f}%). Words are drawn out; the line loses momentum."))
            continue
        k, dur, excess, st = g[0]
        w = wp[k]
        regions.append(Region("stutter", st, w.end, w.word, min(1.0, excess / 0.9), excess / 0.1,
            f"{dur:.2f} s of voiced sound before \"{w.word}\" that is not in the transcript "
            f"({excess:.2f} s more than the baseline at the same word). Typical of a repeated onset or a filler."))

    # ---- clean-up: one cause, one region
    def overlaps(r, o):
        return min(r.end, o.end) - max(r.start, o.start) > 0
    stut = [r for r in regions if r.kind == "stutter"]
    drag = [r for r in regions if r.kind == "dragging"]
    keep = []
    for r in regions:
        dur = r.end - r.start
        if r.kind == "dragging" and r.explanation.startswith("Phrase") and any(
                o is not r and o.kind == "dragging" and overlaps(r, o) for o in regions):
            continue  # the rate detector already reported this phrase
        if r.kind in ("mumble",) and dur < 0.5:
            continue
        if r.kind == "monotone" and dur < 0.6:
            continue
        if r.kind in ("rushed", "dragging") and dur < 0.6:
            continue  # a single short word: too few syllables for a rate estimate
        if r.kind in ("rushed", "dragging") and any(overlaps(r, s) for s in stut):
            continue  # a repeated onset makes the word look slow; the stutter is the cause
        if r.kind == "long_pause" and any(overlaps(r, s) for s in stut):
            continue  # the gaps between repeated onsets are part of the stutter
        if r.kind == "long_pause" and any(d.start <= r.start and r.end <= d.end for d in drag):
            continue  # longer gaps inside a dragged phrase are part of the dragging
        keep.append(r)
    regions = sorted(keep, key=lambda r: r.start)
    return {
        "regions": regions,
        "base_frames": fb, "part_frames": fp, "base_words": wb, "part_words": wp,
        "duration": part.wav.size / SR,
    }


def as_dicts(regions: list[Region]) -> list[dict]:
    return [{**asdict(r), "label": KIND_LABEL[r.kind], "criterion": CRITERION[r.kind]} for r in regions]

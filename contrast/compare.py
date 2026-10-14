"""Baseline vs participant: ONE word-level delta table -> profile, flaws, clarity, heatmap.

Both recordings are force-aligned to the SAME transcript (MMS_FA), so word k of one is
word k of the other. Everything downstream is derived from a single per-word delta table:

    deltas  (per word)       rate log-ratio, extra pause, pitch-movement ratio, energy dB,
                             MFCC-DTW articulation distance, extra sound before the word
       │
       ├── profile           the participant's GLOBAL style vs the reference (median of
       │                     each delta): overall tempo, pause length, pitch range
       ├── regions           LOCAL deviations measured against that profile, so a
       │                     naturally slower speaker is not flagged everywhere; every
       │                     region keeps its word span [i..j] and the numbers behind it
       └── word_heat         the same deltas per word, 0..1, for the transcript heatmap

coach.py, rubric.py and report.py only read this result; nothing is recomputed.
Each deviation is expressed in natural units (syl/s, s, semitones, dB) and as a z-score
against the measurement noise of this pair (robust SD of the word-by-word differences).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import librosa
import numpy as np
import torch

from src import align as A

from . import features as F
from . import text_en as T

SR = 16000

# thresholds: a natural-unit effect size AND |z| must both pass (calibrated in stress_test)
RATE_RATIO = 1.18  # >18% faster/slower than the participant's own overall tempo
PAUSE_DELTA = 0.45  # s of silence beyond the reference pause x the participant's pause style
PITCH_SD_RATIO = 0.62  # pitch movement below 62% of the reference (after pitch-range style)
ENERGY_DB = -5.0  # quieter than the reference, each re its own median
CENTROID_RATIO = 0.78  # spectral centroid drop = muffled articulation
EXTRA_SOUND = 0.15  # s of voiced sound the transcript does not contain
CLARITY_Z = 5.0  # MFCC-DTW distance this many robust SDs above the pair's median
CLARITY_RATIO = 1.8  # and at least 45% above the median word distance
Z_MIN = 1.5

KIND_LABEL = {
    "rushed": "Rushed pace", "dragging": "Dragging pace", "long_pause": "Unplanned long pause",
    "monotone": "Monotone delivery", "mumble": "Quiet / muffled", "stutter": "Disfluency (repeat/stutter)",
    "unclear": "Unclear articulation",
}
CRITERION = {
    "rushed": "Pace", "dragging": "Pace", "long_pause": "Pausing", "monotone": "Expressiveness",
    "mumble": "Volume & clarity", "unclear": "Volume & clarity", "stutter": "Fluency",
}
HEAT_DIMS = ["pace", "pause", "pitch", "volume", "clarity"]


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
    i: int = 0  # first word index (same in both recordings)
    j: int = 0  # last word index
    data: dict = field(default_factory=dict)  # the numbers, for coaching + report


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
    display = [d or s.word for (d, _), s in zip(pairs, real)]
    for d, s in zip(display, real):
        s.word = d
    return Aligned(real, stars, display, wav)


def _robust_sd(x: np.ndarray, floor: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return floor
    mad = float(np.median(np.abs(x - np.median(x)))) * 1.4826
    return max(mad, floor)


def _runs(flags: list[tuple[int, float, float]]) -> list[tuple[int, int, float, float]]:
    """Merge consecutive flagged word indices -> [(i, j, mean_mag, max|z|)]."""
    out, cur = [], []
    for f in sorted(flags):
        if cur and f[0] != cur[-1][0] + 1:
            out.append(cur)
            cur = []
        cur.append(f)
    if cur:
        out.append(cur)
    return [(c[0][0], c[-1][0], float(np.mean([m for _, m, _ in c])), float(max((z for _, _, z in c), key=abs)))
            for c in out]


def _speaking(ws: list, i: int, j: int) -> float:
    """Time from word i to word j minus silences > 0.2 s (articulation time)."""
    return ws[j].end - ws[i].start - sum(max(0.0, ws[k].start - ws[k - 1].end - 0.2) for k in range(i + 1, j + 1))


def _join(ws: list, i: int, j: int) -> str:
    return " ".join(w.word for w in ws[i:j + 1])


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Length-normalized DTW distance between two MFCC sequences (13 x n)."""
    if a.shape[1] < 3 or b.shape[1] < 3:
        return float("nan")
    # drop c0 (loudness) and normalize per word: compare the SHAPE of the sounds only, so a
    # global change elsewhere in the file (inserted room tone, gain) cannot move this word
    a, b = a[1:], b[1:]
    a = (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-6)
    b = (b - b.mean(axis=1, keepdims=True)) / (b.std(axis=1, keepdims=True) + 1e-6)
    D, wp = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    return float(D[-1, -1] / len(wp))


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x))) if np.isfinite(x) else 0.0


def compare(base: Aligned, part: Aligned) -> dict:
    fb, fp = F.frames(base.wav), F.frames(part.wav)
    wb, wp = F.word_features(fb, base.words), F.word_features(fp, part.words)
    n = len(wb)

    # ================================================================ delta table
    lr = np.log(np.array([p.rate / b.rate for b, p in zip(wb, wp)]))
    pb = np.array([w.pause_before for w in wb])
    pp = np.array([w.pause_before for w in wp])
    sb = np.array([w.pitch_sd for w in wb])
    spv = np.array([w.pitch_sd for w in wp])
    eb, ep = np.array([w.energy for w in wb]), np.array([w.energy for w in wp])
    cb, cp = np.array([w.centroid for w in wb]), np.array([w.centroid for w in wp])
    de, cr = ep - eb, cp / cb
    dtw = np.array([_dtw_distance(fb.mfcc[:, F._sl(fb, b.start, b.end)], fp.mfcc[:, F._sl(fp, p.start, p.end)])
                    for b, p in zip(wb, wp)])

    # ================================================================ ① profile
    # The participant's global style, as medians of the deltas. Local flaws are then
    # judged against this style, not against the reference's absolute values.
    tempo = float(np.exp(np.nanmedian(lr)))
    real_gaps = pb > 0.12
    pause_ratio = float(np.clip(np.nanmedian(pp[real_gaps] / pb[real_gaps]), 0.5, 2.5)) if real_gaps.sum() >= 3 else 1.0
    moving = np.isfinite(spv / sb) & (sb > 1.0)
    pitch_ratio = float(np.clip(np.nanmedian((spv / sb)[moving]), 0.4, 1.6)) if moving.sum() >= 3 else 1.0
    clar_med = float(np.nanmedian(dtw))
    profile = {
        "tempo_ratio": round(tempo, 3), "pause_ratio": round(pause_ratio, 3), "pitch_range_ratio": round(pitch_ratio, 3),
        "f0_median_hz": {"reference": round(fb.f0_median_hz, 1), "participant": round(fp.f0_median_hz, 1)},
        "duration_s": {"reference": round(base.wav.size / SR, 2), "participant": round(part.wav.size / SR, 2)},
        "clarity_median": round(clar_med, 2),
    }
    lr_c = lr - np.log(tempo)  # local pace relative to the participant's own tempo
    ratio_c = (spv / sb) / pitch_ratio

    regions: list[Region] = []

    # ---- pace
    sd = _robust_sd(lr_c, 0.05)
    for kind, sign in (("rushed", 1), ("dragging", -1)):
        flags = [(i, min(1.0, abs(lr_c[i]) / np.log(2)), lr_c[i] / sd) for i in range(n)
                 if sign * lr_c[i] > np.log(RATE_RATIO) and sign * lr_c[i] / sd > Z_MIN]
        for i, j, mag, z in _runs(flags):
            rb = sum(w.syl for w in wb[i:j + 1]) / max(1e-3, _speaking(wb, i, j))
            rp = sum(w.syl for w in wp[i:j + 1]) / max(1e-3, _speaking(wp, i, j))
            rel = rp / (rb * tempo)
            if sign * np.log(rel) < np.log(RATE_RATIO):
                continue  # word-level noise that does not hold over the whole region
            style = f" Relative to your own overall tempo: {(rel - 1) * 100:+.0f}%." if abs(tempo - 1) > 0.05 else ""
            regions.append(Region(kind, wp[i].start, wp[j].end, _join(wp, i, j), mag, z,
                f"Speech rate {rp:.1f} syl/s vs baseline {rb:.1f} syl/s ({(rp / rb - 1) * 100:+.0f}%, z={z:+.1f}).{style} "
                + ("Words are compressed; listeners lose the content." if kind == "rushed"
                   else "Words are drawn out; the line loses momentum."),
                i, j, {"rp": rp, "rb": rb, "target": rb * tempo}))

    # ---- pausing (expected pause = reference pause x participant's pause style)
    floor_p = float(np.nanpercentile(fp.energy_db, 5))
    extra_pause = pp - pb * pause_ratio
    sdp = _robust_sd(extra_pause, 0.12)
    for i in range(1, n):
        d = float(extra_pause[i])
        z = d / sdp
        gap = fp.energy_db[F._sl(fp, wp[i - 1].end, wp[i].start)]
        # quiet = near this recording's own noise floor; a "gap" with speech energy in it is
        # mis-aligned quiet speech, not a pause
        quiet = gap.size == 0 or float(np.mean(gap > floor_p + 10)) < 0.3
        if d > PAUSE_DELTA and z > Z_MIN and quiet:
            regions.append(Region("long_pause", wp[i - 1].end, wp[i].start, f"{wp[i-1].word} … {wp[i].word}",
                min(1.0, d / 2.5), z,
                f"Silence of {pp[i]:.2f} s vs {pb[i]:.2f} s in the baseline (+{pp[i] - pb[i]:.2f} s, z={z:+.1f}) "
                f"between \"{wp[i-1].word}\" and \"{wp[i].word}\". A pause this long reads as a lost line, not emphasis.",
                i - 1, i, {"pp": float(pp[i]), "pb": float(pb[i]), "target": float(pb[i] * pause_ratio)}))

    # ---- expressiveness
    sds = _robust_sd((spv - sb * pitch_ratio), 0.2)
    flags = [(i, min(1.0, 1 - ratio_c[i]), (spv[i] - sb[i] * pitch_ratio) / sds) for i in range(n)
             if np.isfinite(ratio_c[i]) and sb[i] > 1.0 and ratio_c[i] < PITCH_SD_RATIO
             and (sb[i] * pitch_ratio - spv[i]) / sds > Z_MIN]
    for i, j, mag, z in _runs(flags):
        b_, p_ = float(np.nanmean(sb[i:j + 1])), float(np.nanmean(spv[i:j + 1]))
        # the word the reference stresses most: highest mean pitch in the region
        pm = [w.pitch_mean if np.isfinite(w.pitch_mean) else -99 for w in wb[i:j + 1]]
        stress = wb[i + int(np.argmax(pm))].word
        regions.append(Region("monotone", wp[i].start, wp[j].end, _join(wp, i, j), mag, z,
            f"Pitch movement {p_:.1f} st vs baseline {b_:.1f} st ({(p_ / b_ - 1) * 100:+.0f}%, z={z:+.1f}). "
            "Intonation is flat, so key words get no emphasis.",
            i, j, {"p": p_, "b": b_, "stress_word": stress}))

    # ---- volume
    sde = _robust_sd(de - np.nanmedian(de), 1.0)
    flags = [(i, min(1.0, max(-de[i] / 20, (1 - cr[i]) / 0.6 if cr[i] < CENTROID_RATIO else 0)), de[i] / sde)
             for i in range(n) if np.isfinite(de[i]) and de[i] < ENERGY_DB and de[i] / sde < -Z_MIN]
    for i, j, mag, z in _runs(flags):
        d_, c_ = float(np.nanmean(de[i:j + 1])), float(np.nanmean(cr[i:j + 1]))
        regions.append(Region("mumble", wp[i].start, wp[j].end, _join(wp, i, j), mag, z,
            f"Energy {d_:+.1f} dB vs baseline (z={z:+.1f}); spectral centroid {(c_ - 1) * 100:+.0f}% "
            f"({float(np.nanmean(cp[i:j+1])):.0f} vs {float(np.nanmean(cb[i:j+1])):.0f} Hz). "
            "Quieter and duller: consonants blur and words are hard to hear.",
            i, j, {"db": d_, "centroid": c_}))

    # ---- articulation clarity: MFCC-DTW distance of each word to the same word in the reference
    sdc = _robust_sd(dtw - clar_med, 0.5)
    zc = (dtw - clar_med) / sdc
    flags = [(i, _clip01((dtw[i] / clar_med - 1) / 1.5), float(zc[i])) for i in range(n)
             if np.isfinite(zc[i]) and zc[i] > CLARITY_Z and dtw[i] > CLARITY_RATIO * clar_med]
    for i, j, mag, z in _runs(flags):
        regions.append(Region("unclear", wp[i].start, wp[j].end, _join(wp, i, j), mag, z,
            f"Spectral shape (MFCC, DTW-aligned) differs from the reference by {float(np.nanmean(dtw[i:j+1])):.1f} "
            f"vs a typical {clar_med:.1f} for this pair (z={z:+.1f}), while pace and volume are normal. "
            "The sounds of the word are not formed the way the reference forms them.",
            i, j, {"dtw": float(np.nanmean(dtw[i:j + 1])), "median": clar_med}))

    # ---- fluency: voiced sound before a word that the transcript does not contain
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
    extra_sound = np.zeros(n)
    cand = []
    for k, (dur, vf, st) in xp.items():
        bd, bv, _ = xb.get(k, (0.0, 0.0, 0.0))
        excess = dur * vf - bd * bv
        extra_sound[k] = max(0.0, excess)
        if excess >= EXTRA_SOUND and vf >= 0.35 and dur * vf >= 1.5 * bd * bv:
            cand.append((k, dur, excess, st))
    groups, cur = [], []
    for c in sorted(cand):
        if cur and c[0] > cur[-1][0] + 1:
            groups.append(cur)
            cur = []
        cur.append(c)
    if cur:
        groups.append(cur)
    for g in groups:
        # extra sound before SEVERAL words, or one long block, = a drawn-out phrase;
        # a short burst before ONE word = a repeated onset / filler
        if len(g) >= 2 or g[0][1] > 1.2:
            i = max(0, g[0][0] - 3 if len(g) == 1 else g[0][0] - 1)
            j = min(n - 1, g[-1][0] + (1 if len(g) == 1 else 0))
            rb = sum(w.syl for w in wb[i:j + 1]) / max(1e-3, wb[j].end - wb[i].start)
            rp = sum(w.syl for w in wp[i:j + 1]) / max(1e-3, wp[j].end - wp[i].start)
            if rp < rb * tempo / RATE_RATIO:
                regions.append(Region("dragging", wp[i].start, wp[j].end, _join(wp, i, j),
                    min(1.0, abs(np.log(rp / rb)) / np.log(2)), -sum(x[2] for x in g) / 0.1,
                    f"Phrase takes {wp[j].end - wp[i].start:.2f} s vs {wb[j].end - wb[i].start:.2f} s in the baseline "
                    f"({rp:.1f} vs {rb:.1f} syl/s, {(rp / rb - 1) * 100:+.0f}%). Words are drawn out; the line loses momentum.",
                    i, j, {"rp": rp, "rb": rb, "target": rb * tempo, "phrase": True}))
            continue
        k, dur, excess, st = g[0]
        keep_frac = (wp[k].end - wp[k].start) / max(1e-3, wb[k].end - wb[k].start)
        if keep_frac < 0.5 and wb[k].end - wb[k].start > 0.15:
            # the aligner could fit only a sliver of the word: the sound in its place does not
            # match the word = mispronounced / unclear, not a repeat (a repeat keeps the word whole)
            regions.append(Region("unclear", st, wp[k].end, wp[k].word, min(1.0, 1 - keep_frac), (1 - keep_frac) * 5,
                f"Only {keep_frac * 100:.0f}% of \"{wp[k].word}\" matched the expected sounds "
                f"({wp[k].end - wp[k].start:.2f} s vs {wb[k].end - wb[k].start:.2f} s in the reference); the other "
                f"{dur:.2f} s did not sound like the word. Typical of a mispronounced or swallowed word.",
                k, k, {"keep_frac": keep_frac}))
            continue
        regions.append(Region("stutter", st, wp[k].end, wp[k].word, min(1.0, excess / 0.9), excess / 0.1,
            f"{dur:.2f} s of voiced sound before \"{wp[k].word}\" that is not in the transcript "
            f"({excess:.2f} s more than the baseline at the same word). Typical of a repeated onset or a filler.",
            k, k, {"extra": excess}))

    # ================================================================ clean-up: one cause, one region
    def overlaps(r, o):
        return min(r.end, o.end) - max(r.start, o.start) > 0
    by = {k: [r for r in regions if r.kind == k] for k in KIND_LABEL}
    keep = []
    for r in regions:
        dur = r.end - r.start
        if r.kind == "dragging" and r.data.get("phrase") and any(
                o is not r and overlaps(r, o) for o in by["dragging"]):
            continue  # the rate detector already reported this phrase
        if r.kind == "mumble" and dur < 0.5:
            continue
        if r.kind == "monotone" and dur < 0.6:
            continue
        if r.kind in ("rushed", "dragging") and dur < 0.6:
            continue  # a single short word: too few syllables for a rate estimate
        if r.kind in ("rushed", "dragging", "long_pause") and any(overlaps(r, s) for s in by["stutter"]):
            continue  # a repeated onset looks slow and gappy; the stutter is the cause
        if r.kind == "long_pause" and any(d.start <= r.start and r.end <= d.end for d in by["dragging"]):
            continue  # longer gaps inside a dragged phrase are part of the dragging
        if r.kind == "unclear" and dur < 0.25:
            continue
        keep.append(r)
    # MFCC-based clarity is checked against the regions that SURVIVED: any nearby flaw
    # (time-scaling, pitch resynthesis, muffling) also moves MFCCs, so it is the cause
    others = [o for o in keep if o.kind != "unclear"]
    keep = [r for r in keep if not (r.kind == "unclear" and "dtw" in r.data and any(
        min(r.end, o.end + 1.0) - max(r.start, o.start - 1.0) > 0 for o in others))]
    regions = sorted(keep, key=lambda r: r.start)

    # ================================================================ ⑤ word heat (same deltas)
    heat = []
    for k in range(n):
        dims = {
            "pace": _clip01(abs(lr_c[k]) / np.log(2) * 1.5),
            "pause": _clip01(extra_pause[k] / 2.0) if k else 0.0,
            "pitch": _clip01(1 - ratio_c[k]) if np.isfinite(ratio_c[k]) and sb[k] > 1.0 else 0.0,
            "volume": _clip01(-de[k] / 15),
            "clarity": _clip01(zc[k] / 6) if np.isfinite(zc[k]) else 0.0,
        }
        inside = [r.kind for r in regions if r.i <= k <= r.j]
        heat.append({"word": wp[k].word, "start": wp[k].start, "end": wp[k].end, **{d: round(v, 3) for d, v in dims.items()},
                     "heat": round(max(dims.values()) if not inside else max(0.75, max(dims.values())), 3),
                     "flaw": inside[0] if inside else None})

    return {
        "regions": regions, "profile": profile, "word_heat": heat,
        "base_frames": fb, "part_frames": fp, "base_words": wb, "part_words": wp,
        "base_wav": base.wav, "part_wav": part.wav, "duration": part.wav.size / SR,
    }


def as_dicts(regions: list[Region]) -> list[dict]:
    return [{**asdict(r), "label": KIND_LABEL[r.kind], "criterion": CRITERION[r.kind]} for r in regions]

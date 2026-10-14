"""Inject delivery flaws into a good recording, with exact ground-truth timestamps.

Because we create the flaw ourselves, its start/end in the new recording is known to the
sample, so the labels are as precise as labels can be. Every operation keeps a time map
from the original timeline to the new one, so word timestamps are carried over too and
the flawed file needs no re-annotation.

Flaw types (severity 1 = almost perfect, 3 = clearly bad):

    rushed      region spoken faster (pitch kept, PSOLA)        x0.80 / 0.65 / 0.50 duration
    dragging    region spoken slower (pitch kept, PSOLA)        x1.30 / 1.60 / 2.00 duration
    long_pause  silence inserted between two words              0.8 / 1.6 / 2.8 s
    monotone    pitch contour flattened toward its mean         60% / 85% / 100% flatter
    mumble      quieter and muffled (gain + low-pass)           -8/-14/-20 dB, 3.0/1.8/1.0 kHz
    stutter     word onset repeated                             1 / 2 / 3 repeats
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import parselmouth
from parselmouth.praat import call

SR = 16000
FLAW_TYPES = ["rushed", "dragging", "long_pause", "monotone", "mumble", "stutter"]
LEVELS = {
    "rushed": [0.80, 0.65, 0.50],
    "dragging": [1.30, 1.60, 2.00],
    "long_pause": [0.8, 1.6, 2.8],
    "monotone": [0.60, 0.85, 1.00],
    "mumble": [(-8, 3000), (-14, 1800), (-20, 1000)],
    "stutter": [1, 2, 3],
}
XFADE = int(0.008 * SR)  # 8 ms crossfade at every splice, no clicks


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class FlawLabel:
    type: str
    severity: int
    start: float  # seconds, in the FLAWED recording
    end: float
    words: str
    param: float | str


@dataclass
class Piece:
    """One stretch of output audio and where it came from in the original."""
    audio: np.ndarray
    src_start: float  # original timeline; inserted audio has src_start == src_end
    src_end: float
    labels: list[FlawLabel] = field(default_factory=list)


# ------------------------------------------------------------------ helpers


def _snd(x: np.ndarray) -> parselmouth.Sound:
    return parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR)


def _np(s: parselmouth.Sound) -> np.ndarray:
    return s.values[0].astype(np.float32)


def room_tone(wav: np.ndarray, dur: float) -> np.ndarray:
    """Quietest 200 ms of the recording, tiled: inserted pauses sound like the room."""
    win = int(0.2 * SR)
    hop = win // 2
    best, best_rms = 0, np.inf
    for i in range(0, max(1, wav.size - win), hop):
        r = float(np.sqrt(np.mean(wav[i:i + win] ** 2)))
        if r < best_rms:
            best, best_rms = i, r
    tone = wav[best:best + win]
    n = int(dur * SR)
    reps = int(np.ceil(n / max(1, tone.size)))
    return np.tile(tone, reps)[:n].astype(np.float32)


def stretch(x: np.ndarray, factor: float) -> np.ndarray:
    """Duration x factor, pitch unchanged (Praat overlap-add / PSOLA)."""
    return _np(call(_snd(x), "Lengthen (overlap-add)", 75, 600, float(factor)))


def flatten_pitch(x: np.ndarray, alpha: float) -> np.ndarray:
    """Pull the F0 contour toward its mean by `alpha` (1.0 = completely flat)."""
    snd = _snd(x)
    manip = call(snd, "To Manipulation", 0.01, 75, 600)
    tier = call(manip, "Extract pitch tier")
    n = call(tier, "Get number of points")
    if n < 2:
        return x
    mean = call(tier, "Get mean (points)", 0, 0)
    call(tier, "Formula", f"{mean} + (self - {mean}) * {1 - alpha}")
    call([tier, manip], "Replace pitch tier")
    out = _np(call(manip, "Get resynthesis (overlap-add)"))
    return out[: x.size] if out.size >= x.size else np.pad(out, (0, x.size - out.size))


def muffle(x: np.ndarray, gain_db: float, cutoff: float) -> np.ndarray:
    snd = _snd(x)
    filt = call(snd, "Filter (pass Hann band)", 0, float(cutoff), 300)
    return (_np(filt) * 10 ** (gain_db / 20)).astype(np.float32)


def _join(parts: list[np.ndarray]) -> np.ndarray:
    out = parts[0].astype(np.float32)
    ramp = np.linspace(0, 1, XFADE, dtype=np.float32)
    for p in parts[1:]:
        p = p.astype(np.float32)
        if out.size > XFADE and p.size > XFADE:
            mixed = out[-XFADE:] * (1 - ramp) + p[:XFADE] * ramp
            out = np.concatenate([out[:-XFADE], mixed, p[XFADE:]])
        else:
            out = np.concatenate([out, p])
    return out


# ------------------------------------------------------------------- engine


def _cut_points(words: list[Word], i: int, j: int, total: float) -> tuple[float, float]:
    """Region [words i..j] in the original, cut in the middle of the neighbouring pauses."""
    a = (words[i - 1].end + words[i].start) / 2 if i > 0 else max(0.0, words[i].start - 0.05)
    b = (words[j].end + words[j + 1].start) / 2 if j + 1 < len(words) else min(total, words[j].end + 0.05)
    return a, b


def apply_flaws(wav: np.ndarray, words: list[Word], flaws: list[tuple[str, int, int, int]]):
    """flaws: [(type, severity 1-3, first word idx, last word idx)], non-overlapping.

    Returns (flawed wav, words with new timestamps, [FlawLabel]).
    """
    total = wav.size / SR
    flaws = sorted(flaws, key=lambda f: f[2])
    pieces: list[Piece] = []
    cursor = 0.0
    for ftype, sev, i, j in flaws:
        p = LEVELS[ftype][sev - 1]
        a, b = _cut_points(words, i, j, total)
        if ftype == "long_pause":
            # silence goes into the gap after word i (j is ignored)
            a = b = (words[i].end + words[i + 1].start) / 2
        if a > cursor:
            pieces.append(Piece(wav[int(cursor * SR):int(a * SR)], cursor, a))
        seg = wav[int(a * SR):int(b * SR)]
        text = " ".join(w.word for w in words[i:j + 1])
        lab = FlawLabel(ftype, sev, 0, 0, text, p if not isinstance(p, tuple) else f"{p[0]} dB, {p[1]} Hz")
        if ftype in ("rushed", "dragging"):
            pieces.append(Piece(stretch(seg, p), a, b, [lab]))
        elif ftype == "monotone":
            pieces.append(Piece(flatten_pitch(seg, p), a, b, [lab]))
        elif ftype == "mumble":
            pieces.append(Piece(muffle(seg, *p), a, b, [lab]))
        elif ftype == "long_pause":
            lab.words = f"{words[i].word} … {words[i + 1].word}"
            pieces.append(Piece(room_tone(wav, p), a, a, [lab]))
        elif ftype == "stutter":
            w = words[i]
            onset = wav[int(w.start * SR):int(min(w.end, w.start + max(0.12, (w.end - w.start) * 0.4)) * SR)]
            gap = room_tone(wav, 0.12)
            rep = []
            for _ in range(int(p)):
                rep += [onset, gap]
            ws = w.start
            if ws > a:
                pieces.append(Piece(wav[int(a * SR):int(ws * SR)], a, ws))
            lab.words = w.word
            pieces.append(Piece(_join(rep), ws, ws, [lab]))
            a = ws
            seg = wav[int(a * SR):int(b * SR)]
            pieces.append(Piece(seg, a, b))
        cursor = b
    if cursor < total:
        pieces.append(Piece(wav[int(cursor * SR):], cursor, total))

    # assemble + time map
    out_parts, labels, mapping = [], [], []  # mapping: (src_a, src_b, new_a, new_b)
    t = 0.0
    for k, pc in enumerate(pieces):
        d = pc.audio.size / SR
        if k > 0 and pc.audio.size > XFADE:
            t -= XFADE / SR  # _join overlaps each piece with the previous by XFADE
        mapping.append((pc.src_start, pc.src_end, t, t + d))
        for lab in pc.labels:
            lab.start, lab.end = round(t, 3), round(t + d, 3)
            labels.append(lab)
        out_parts.append(pc.audio)
        t += d
    out = _join(out_parts)

    def remap(x: float) -> float:
        for sa, sb, na, nb in mapping:
            if sb > sa and sa <= x <= sb:
                return na + (x - sa) / (sb - sa) * (nb - na)
        return x

    new_words = [Word(w.word, round(remap(w.start), 3), round(remap(w.end), 3)) for w in words]
    # a stutter's label should end where the real word ends
    for lab in labels:
        if lab.type == "stutter":
            nxt = next((w for w in new_words if w.start >= lab.end - 0.02), None)
            if nxt:
                lab.end = round(nxt.end, 3)
    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 0.99:
        out = out / peak * 0.99
    return out.astype(np.float32), new_words, labels


def pick_region(words: list[Word], ftype: str, key: str, n_words: int = 7) -> tuple[int, int]:
    """Deterministic region choice (same place for every severity of one flaw type)."""
    h = int(hashlib.sha1(f"{key}|{ftype}".encode()).hexdigest(), 16)
    lo, hi = 2, max(3, len(words) - n_words - 2)
    i = lo + h % max(1, hi - lo)
    if ftype in ("long_pause", "stutter"):
        return i, i
    return i, min(len(words) - 1, i + n_words - 1)

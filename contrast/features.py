"""Speaker-normalized acoustic features on a 10 ms grid and per word.

Speaker agnostic by construction:
  - pitch in SEMITONES relative to the speaker's own median F0 (a deep and a high voice
    both sit around 0 st; only the movement counts)
  - energy in dB relative to the speaker's median speech energy
  - speech rate as syllables per second over a 3-word window
Frame features: F0 (Praat autocorrelation), intensity (Praat), HNR (Praat cross-
correlation, a voice-clarity proxy), spectral centroid and 13 MFCCs (librosa, via FFT).
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import parselmouth

from .text_en import syllables

SR = 16000
HOP = 0.01  # 10 ms


@dataclass
class Frames:
    t: np.ndarray  # seconds
    f0_hz: np.ndarray  # nan = unvoiced
    pitch_st: np.ndarray  # semitones re speaker median, nan = unvoiced
    energy_db: np.ndarray  # dB re speaker median speech energy
    hnr_db: np.ndarray  # nan = unvoiced
    centroid_hz: np.ndarray
    mfcc: np.ndarray  # (13, n)
    f0_median_hz: float


@dataclass
class WordFeat:
    word: str
    start: float
    end: float
    syl: int
    pause_before: float
    rate: float  # syllables / s, 3-word window
    pitch_mean: float  # st
    pitch_sd: float  # st, 5-word window (expressiveness)
    energy: float  # dB re median
    hnr: float
    centroid: float


def _on_grid(xs: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if xs.size == 0:
        return np.full(grid.size, np.nan)
    return np.interp(grid, xs, vals, left=np.nan, right=np.nan)


def frames(wav: np.ndarray) -> Frames:
    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=SR)
    n = int(len(wav) / SR / HOP)
    grid = np.arange(n) * HOP + HOP / 2

    pitch = snd.to_pitch_ac(time_step=HOP, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 <= 0] = np.nan
    # nearest frame, not interpolation: interpolating across unvoiced gaps invents pitch
    idx = np.clip(np.searchsorted(pitch.xs(), grid), 0, max(0, f0.size - 1))
    f0g = f0[idx] if f0.size else np.full(n, np.nan)
    med = float(np.nanmedian(f0g)) if np.isfinite(f0g).any() else 150.0
    st = 12 * np.log2(f0g / med)

    inten = snd.to_intensity(minimum_pitch=75, time_step=HOP)
    e = _on_grid(inten.xs(), inten.values[0], grid)
    speech = e > np.nanpercentile(e, 40)  # rough speech mask for the reference level
    e_rel = e - float(np.nanmedian(e[speech])) if speech.any() else e - np.nanmedian(e)

    harm = snd.to_harmonicity_cc(time_step=HOP, minimum_pitch=75)
    h = harm.values[0].astype(float)
    h[h < -100] = np.nan
    hg = h[np.clip(np.searchsorted(harm.xs(), grid), 0, max(0, h.size - 1))] if h.size else np.full(n, np.nan)

    hop = int(HOP * SR)
    cen = librosa.feature.spectral_centroid(y=wav, sr=SR, n_fft=512, hop_length=hop)[0]
    mf = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=13, n_fft=512, hop_length=hop)
    mf = mf - mf.mean(axis=1, keepdims=True)  # cepstral mean normalization (channel/speaker)

    def fit(a):
        a = a[..., :n]
        if a.shape[-1] < n:
            pad = [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])]
            a = np.pad(a, pad, mode="edge")
        return a

    return Frames(grid, f0g, st, e_rel, hg, fit(cen), fit(mf), med)


def _sl(fr: Frames, a: float, b: float) -> slice:
    return slice(max(0, int(a / HOP)), max(int(a / HOP) + 1, int(b / HOP)))


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def word_features(fr: Frames, words: list) -> list[WordFeat]:
    """`words`: objects with .word .start .end (seconds)."""
    out: list[WordFeat] = []
    n = len(words)
    for i, w in enumerate(words):
        s = _sl(fr, w.start, w.end)
        a, b = max(0, i - 1), min(n - 1, i + 1)
        # articulation rate: silences > 0.2 s inside the window are not "slow speech"
        span = words[b].end - words[a].start - sum(
            max(0.0, words[k].start - words[k - 1].end - 0.2) for k in range(a + 1, b + 1))
        syl = sum(syllables(x.word) for x in words[a:b + 1])
        a5, b5 = max(0, i - 2), min(n - 1, i + 2)
        win = fr.pitch_st[_sl(fr, words[a5].start, words[b5].end)]
        voiced = win[np.isfinite(win)]
        out.append(WordFeat(
            word=w.word, start=w.start, end=w.end, syl=syllables(w.word),
            pause_before=max(0.0, w.start - words[i - 1].end) if i else 0.0,
            rate=syl / span if span > 0 else float("nan"),
            pitch_mean=_nanmean(fr.pitch_st[s]),
            pitch_sd=float(np.std(voiced)) if voiced.size > 5 else float("nan"),
            energy=_nanmean(fr.energy_db[s]),
            hnr=_nanmean(fr.hnr_db[s]),
            centroid=_nanmean(fr.centroid_hz[s]),
        ))
    return out

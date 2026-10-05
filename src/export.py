"""Eksport — LJSpeech formatida dataset yozish.

Chiqish tuzilishi:
    out/
      wavs/0001.wav ...
      metadata.csv          0001|matn
      rejected.csv          sabab bilan rad etilganlar
      report.json           statistika

Tadqiqotdan:
  - Masterdan BITTA qadamda resample, soxr_vhq sifati bilan
  - 22050 Hz -> VITS/LJSpeech · 24000 Hz -> XTTS/StyleTTS2
  - Sukunatni nolga kesib, keyin HAR FAYLGA BIR XIL pad (StyleTTS2).
    Pad miqdori emas, MUTANOSIBLIGI muhim — nomutanosib trim bo'lsa model
    nafas/shovqin chiqarishni o'rganadi.
  - Loudness butun fayl bo'yicha bir marta (-23 LUFS), klip-klip emas:
    har klipni alohida normallashtirish ovoz dinamikasini buzadi.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf

TAIL_PAD = 0.10  # 100 ms, har faylga bir xil (StyleTTS2)
TARGET_LUFS = -23.0  # EBU R128
TRUE_PEAK_MAX = -1.0  # dBTP


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    try:
        import soxr

        return soxr.resample(wav, sr_in, sr_out, quality="VHQ")
    except ImportError:
        import librosa

        return librosa.resample(wav, orig_sr=sr_in, target_sr=sr_out, res_type="soxr_vhq")


def normalize_loudness(wav: np.ndarray, sr: int, target: float = TARGET_LUFS) -> np.ndarray:
    """Butun fayl uchun bir marta chaqirilsin, klip-klip emas."""
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        if not np.isfinite(loudness):
            return wav
        out = pyln.normalize.loudness(wav, loudness, target)
    except Exception:
        return wav
    # true peak cheklovi
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    limit = 10 ** (TRUE_PEAK_MAX / 20)
    if peak > limit:
        out = out * (limit / peak)
    return out


def trim_and_pad(clip: np.ndarray, sr: int, tail_pad: float = TAIL_PAD) -> np.ndarray:
    """Chetlardagi jimlikni kesib, oxiriga bir xil pad qo'shadi."""
    if clip.size == 0:
        return clip
    thresh = max(1e-4, float(np.max(np.abs(clip))) * 0.01)
    nz = np.where(np.abs(clip) > thresh)[0]
    if nz.size:
        clip = clip[nz[0] : nz[-1] + 1]
    # Boshiga ham, oxiriga ham BIR XIL pad (StyleTTS2: miqdor emas, mutanosiblik muhim).
    # Oldin faqat oxiriga qo'shilardi: boshida 0-3 ms, oxirida 100 ms chiqardi.
    pad = np.zeros(int(tail_pad * sr), dtype=clip.dtype)
    return np.concatenate([pad, clip, pad])


@dataclass
class ExportResult:
    written: int
    rejected: int
    total_hours: float
    out_dir: str


def export_ljspeech(
    master_wav: np.ndarray,
    master_sr: int,
    segments: list,  # list[Segment]
    out_dir: str,
    target_sr: int = 22050,
    prefix: str = "",
    normalize: bool = True,
    extra: dict | None = None,
) -> ExportResult:
    """Segmentlarni LJSpeech formatida yozadi.

    `master_wav` — BUTUN audio (mono float32), `segments` — Segment ro'yxati.
    Rad etilganlar (`reject` yoki `verdict.ok == False`) `rejected.csv` ga tushadi.
    """
    wavs_dir = os.path.join(out_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    if normalize:
        master_wav = normalize_loudness(master_wav, master_sr)
    master_wav = _resample(master_wav, master_sr, target_sr)

    rows: list[tuple[str, str]] = []
    # provenance: har klip manba audioning qaysi joyidan olingani + bahosi.
    # Baholash/debug uchun zarur — metadata.csv da vaqt bo'lmaydi (LJSpeech formati).
    prov: list[tuple[str, str, str, str, str, str]] = []
    rejected: list[tuple[str, str, str, str, str]] = []
    total = 0.0
    n = 0

    for seg in segments:
        reason = getattr(seg, "reject", None)
        verdict = getattr(seg, "verdict", None)
        if reason is None and verdict is not None and not verdict.ok:
            reason = f"{verdict.action}: {verdict.reason}"
        if reason:
            rejected.append((
                f"{seg.start:.3f}-{seg.end:.3f}", reason,
                f"{verdict.cer:.4f}" if verdict is not None and verdict.cer is not None else "",
                f"{getattr(seg, 'mean_score', 0.0):.3f}",
                seg.text,
            ))
            continue

        a = int(seg.start * target_sr)
        b = min(int(seg.end * target_sr), master_wav.size)
        if b <= a:
            rejected.append((f"{seg.start:.3f}-{seg.end:.3f}", "bo'sh bo'lak", "", "", seg.text))
            continue

        clip = trim_and_pad(master_wav[a:b], target_sr)
        n += 1
        name = f"{prefix}{n:04d}"
        sf.write(os.path.join(wavs_dir, name + ".wav"), clip, target_sr, subtype="PCM_16")
        rows.append((name, seg.text))
        v = getattr(seg, "verdict", None)
        prov.append((
            name,
            f"{seg.start:.3f}",
            f"{seg.end:.3f}",
            f"{getattr(seg, 'min_score', 0.0):.3f}",
            f"{v.cer:.4f}" if v is not None and v.cer is not None else "",
            f"{v.snr:.1f}" if v is not None and v.snr is not None else "",
        ))
        total += clip.size / target_sr

    with open(os.path.join(out_dir, "metadata.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for name, text in rows:
            w.writerow([name, text])

    with open(os.path.join(out_dir, "segments.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["nom", "start", "end", "min_score", "cer", "snr_db"])
        w.writerows(prov)

    with open(os.path.join(out_dir, "rejected.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["vaqt", "sabab", "cer", "mean_score", "matn"])
        w.writerows(rejected)

    report = {
        "yozilgan_klip": n,
        "rad_etilgan": len(rejected),
        "jami_soat": round(total / 3600, 4),
        "ortacha_sek": round(total / n, 2) if n else 0.0,
        "sample_rate": target_sr,
        "saqlanish_foizi": round(n / (n + len(rejected)) * 100, 1) if (n + len(rejected)) else 0.0,
    }
    if extra:
        report.update(extra)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return ExportResult(n, len(rejected), total / 3600, out_dir)

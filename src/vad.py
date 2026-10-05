"""silero-VAD — nutq chegaralarini aniq topish.

Nega kerak: CTC alignment chegaralarni nutqning ISHONCHLI YADROSIGA qo'yadi,
boshlanish/tugash chetiga emas. FLEURS uz_uz da o'lchandi: birinchi so'z nutq
boshlanishidan ~290 ms keyin, oxirgi so'z tugashidan ~480 ms oldin.

Ko'r-ko'rona pad qo'shish yarim yechim: 23/62 segmentda hali ham nutq kesilgan edi
(amplituda 0.075 — nutqning 95-protsentilidan balandroq). VAD esa nutq qayerda
BOSHLANISHINI bilib beradi, taxmin qilmaydi.

Parametrlar tadqiqotdan (manbalar `tadqiqot/research_3_quality.md`):
  threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=250-300
  (default 100 juda agressiv), speech_pad_ms NeMo Curator'da 300
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


def speech_regions(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 280,
    speech_pad_ms: int = 0,
) -> list[tuple[float, float]]:
    """Nutq oraliqlari ro'yxati: [(start_sek, end_sek), ...].

    `speech_pad_ms=0` — pad'ni VAD emas, segmenter o'zi boshqaradi.
    Xatolik bo'lsa bo'sh ro'yxat qaytaradi (quvur VAD'siz ishlashda davom etadi).
    """
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except Exception:
        return []

    try:
        model = load_silero_vad()
        ts = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)),
            model,
            sampling_rate=sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
        )
    except Exception:
        return []
    return [(float(t["start"]), float(t["end"])) for t in ts]


def snap(
    start: float,
    end: float,
    regions: list[tuple[float, float]],
    lo: float,
    hi: float,
) -> tuple[float, float]:
    """Chegaralarni VAD nutq oraliqlariga moslashtiradi.

    `lo`/`hi` — qattiq chegaralar (qo'shni so'zga kirmaslik uchun). VAD qiymati
    ulardan tashqariga chiqsa, cheklanadi. Mos oraliq topilmasa — o'zgarmaydi.
    """
    if not regions:
        return start, end

    # start: segment boshlanishini QAMRAB OLGAN yoki undan oldin boshlanadigan
    # eng yaqin oraliqning boshiga tortamiz
    new_start = start
    for r0, r1 in regions:
        if r1 < lo:
            continue
        if r0 > start:
            break
        new_start = max(lo, min(start, r0))
    # end: tugashni qamrab olgan oraliqning oxiriga cho'zamiz
    new_end = end
    for r0, r1 in regions:
        if r0 > hi:
            break
        if r1 >= end:
            new_end = min(hi, max(end, r1))
            break

    if new_end <= new_start:
        return start, end
    return new_start, new_end

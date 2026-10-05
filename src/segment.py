"""Segmentlash — moslashtirilgan so'zlardan TTS kliplarini yasash.

Qoidalar tadqiqotdan (manbalar `tadqiqot/` papkasida):
  - Maqsad uzunlik 2-12 sek (ManaTTS retsepti). LJSpeech realligi: 1.11-10.10 s,
    o'rtacha 6.57 s
  - Gap chegarasida kesish birinchi o'rinda (NFA default: . ? ! ...)
  - So'z o'rtasidan HECH QACHON kesmaslik
  - Pauzani o'rtasidan bo'lish: int((prev.end + next.start) / 2) — MMS get_spans
  - Qo'shni so'zga kirmaslik: pad qo'shni so'z chegarasi bilan cheklanadi
  - Gap < 0.55 s bo'lsa birlashtirish (WenetSpeech4TTS)
  - Ichki uzluksiz sukunat > 930 ms bo'lsa tashlash (GLOBE)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .align import WordSpan
from .uz_text import ends_sentence

# Tadqiqotdan olingan chegaralar
MIN_DUR = 2.0
MAX_DUR = 12.0
HARD_MAX_DUR = 20.0  # bundan uzunini umuman olmaymiz (NeMo MAX_DURATION)
MERGE_GAP = 0.55  # WenetSpeech4TTS
MAX_INNER_SILENCE = 0.93  # GLOBE
# CTC alignment chegaralarni nutqning ISHONCHLI YADROSIGA qo'yadi, chetiga emas.
# FLEURS uz_uz da o'lchandi (2026-10-02): MMS_FA birinchi so'zni nutq boshlanishidan
# ~290 ms KEYIN, oxirgi so'zni tugashidan ~480 ms OLDIN qo'yadi. Shu sababli pad
# NOSIMMETRIK: oxiriga ko'proq. Chunking aybdor emas — chunk 10/20/40s bir xil natija.
# Pad hech qachon qo'shni so'zga kirmaydi (pauzaning o'rtasi bilan cheklangan).
PAD_START = 0.35
PAD_END = 0.55

# Bitta so'z shundan uzoq cho'zilsa — alignment bo'shliqni "yopish" uchun
# so'zni sun'iy cho'zgan. Amalda ASR gallyutsinatsiyasining eng aniq belgisi:
# sinovda 3.16 sekundli "toʻxtang" score 0.001 bilan chiqdi.
# Bu BIZNING chegara (manbadan emas) — o'zbekda eng uzun so'z ~1 sekund.
MAX_WORD_DUR = 1.5


@dataclass
class Segment:
    start: float
    end: float
    text: str  # dataset matni (punktuatsiya bilan)
    words: list[WordSpan] = field(default_factory=list)
    reject: str | None = None  # rad etilgan bo'lsa — sababi

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def min_score(self) -> float:
        return min((w.score for w in self.words), default=0.0)

    @property
    def mean_score(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.score for w in self.words) / len(self.words)

    @property
    def max_word_dur(self) -> float:
        """Eng uzoq cho'zilgan so'z — sun'iy cho'zilish belgisi."""
        return max((w.duration for w in self.words), default=0.0)

    @property
    def max_inner_silence(self) -> float:
        """So'zlar orasidagi eng uzun jimlik."""
        gaps = [
            b.start - a.end for a, b in zip(self.words, self.words[1:]) if b.start > a.end
        ]
        return max(gaps, default=0.0)


def _group_by_sentence(pairs: list[tuple[str, str]], spans: list[WordSpan]) -> list[list[int]]:
    """So'z indekslarini gaplarga bo'ladi."""
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, (ds_word, _) in enumerate(pairs):
        if i >= len(spans):
            break
        cur.append(i)
        if ends_sentence(ds_word):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _duration(idx: list[int], spans: list[WordSpan]) -> float:
    return spans[idx[-1]].end - spans[idx[0]].start


def _split_long(idx: list[int], spans: list[WordSpan]) -> list[list[int]]:
    """MAX_DUR dan uzun guruhni eng katta pauzalardan bo'ladi."""
    if _duration(idx, spans) <= MAX_DUR or len(idx) < 2:
        return [idx]
    # eng katta ichki pauzani topamiz
    best_pos, best_gap = None, -1.0
    for k in range(1, len(idx)):
        gap = spans[idx[k]].start - spans[idx[k - 1]].end
        # chetga juda yaqin bo'lmasin — ikki bo'lak ham MIN_DUR ga yaqin bo'lsin
        if _duration(idx[:k], spans) < MIN_DUR or _duration(idx[k:], spans) < MIN_DUR:
            continue
        if gap > best_gap:
            best_pos, best_gap = k, gap
    if best_pos is None:
        return [idx]  # bo'lib bo'lmaydi
    left, right = idx[:best_pos], idx[best_pos:]
    return _split_long(left, spans) + _split_long(right, spans)


def _merge_short(groups: list[list[int]], spans: list[WordSpan]) -> list[list[int]]:
    """MIN_DUR dan qisqa guruhlarni keyingisiga qo'shadi (gap kichik bo'lsa)."""
    out: list[list[int]] = []
    for g in groups:
        if out:
            prev = out[-1]
            gap = spans[g[0]].start - spans[prev[-1]].end
            too_short = _duration(prev, spans) < MIN_DUR or _duration(g, spans) < MIN_DUR
            would_fit = _duration(prev + g, spans) <= MAX_DUR
            if too_short and gap < MERGE_GAP and would_fit:
                out[-1] = prev + g
                continue
        out.append(g)
    return out


def _bounds(
    idx: list[int], spans: list[WordSpan], total_dur: float
) -> tuple[float, float, float, float]:
    """Klip chegaralari + QATTIQ cheklovlar.

    Qaytaradi: (start, end, lo, hi)
      start/end — pad qo'shilgan chegaralar
      lo/hi     — pauzaning o'rtasi: bundan tashqariga chiqish qo'shni so'zga
                  kirish degani, VAD snap ham bu chegaradan oshmaydi
    """
    first, last = idx[0], idx[-1]
    w_start = spans[first].start
    w_end = spans[last].end

    # chapdagi qattiq chegara — oldingi so'z bilan orasidagi pauzaning o'rtasi
    if first > 0:
        prev_end = spans[first - 1].end
        lo = prev_end + (w_start - prev_end) / 2
    else:
        lo = 0.0
    # o'ngdagi qattiq chegara
    if last + 1 < len(spans):
        next_start = spans[last + 1].start
        hi = w_end + (next_start - w_end) / 2
    else:
        hi = total_dur

    start = max(lo, w_start - PAD_START)
    end = min(hi, w_end + PAD_END)
    return start, end, lo, hi


def build_segments(
    pairs: list[tuple[str, str]],
    spans: list[WordSpan],
    total_dur: float,
    vad_regions: list[tuple[float, float]] | None = None,
    max_inner_silence: float | None = None,
) -> list[Segment]:
    """So'z juftliklari + moslashtirilgan span'lardan segmentlar yasaydi.

    `pairs` — uz_text.split_aligned() natijasi, `spans` — align.align() natijasi
    (yulduzlar olib tashlangan). Indekslari mos bo'lishi SHART.

    Rad etilganlar ham qaytariladi (`reject` to'ldirilgan) — hisobot uchun.
    """
    if not spans:
        return []
    # GLOBE'ning 0.93 s chegarasi oddiy o'qish uchun. Badiiy o'qishda (O'tkan kunlar,
    # Malika Ibrohimova) dramatik pauzalar 1.0-1.4 s — 13/82 segment shu sababli
    # rad etilgan edi. Material turiga qarab CLI'dan (--max-pause) beriladi.
    max_sil = MAX_INNER_SILENCE if max_inner_silence is None else max_inner_silence

    groups = _group_by_sentence(pairs, spans)
    expanded: list[list[int]] = []
    for g in groups:
        expanded.extend(_split_long(g, spans))
    groups = _merge_short(expanded, spans)

    out: list[Segment] = []
    for idx in groups:
        start, end, lo, hi = _bounds(idx, spans, total_dur)
        if vad_regions:
            # VAD nutq qayerda boshlanishini BILADI — pad esa taxmin qiladi.
            # Qattiq chegaralar (lo/hi) saqlanadi: qo'shni so'zga kirilmaydi.
            from .vad import snap

            start, end = snap(start, end, vad_regions, lo, hi)
        text = " ".join(pairs[i][0] for i in idx)
        seg = Segment(start=start, end=end, text=text, words=[spans[i] for i in idx])

        if seg.duration < MIN_DUR:
            seg.reject = f"juda qisqa ({seg.duration:.2f}s < {MIN_DUR})"
        elif seg.duration > HARD_MAX_DUR:
            seg.reject = f"juda uzun ({seg.duration:.2f}s > {HARD_MAX_DUR})"
        elif seg.max_inner_silence > max_sil:
            seg.reject = f"ichki sukunat {seg.max_inner_silence:.2f}s > {max_sil}"
        elif seg.max_word_dur > MAX_WORD_DUR:
            seg.reject = f"so'z sun'iy cho'zilgan ({seg.max_word_dur:.2f}s > {MAX_WORD_DUR})"
        out.append(seg)

    return out


def summary(segments: list[Segment]) -> dict:
    kept = [s for s in segments if s.reject is None]
    total = sum(s.duration for s in kept)
    rejects: dict[str, int] = {}
    for s in segments:
        if s.reject:
            key = s.reject.split("(")[0].split(">")[0].strip()
            rejects[key] = rejects.get(key, 0) + 1
    durs = sorted(s.duration for s in kept)
    return {
        "jami_segment": len(segments),
        "qabul": len(kept),
        "rad": len(segments) - len(kept),
        "rad_sabablari": rejects,
        "jami_soat": round(total / 3600, 3),
        "ortacha_sek": round(total / len(kept), 2) if kept else 0.0,
        "min_sek": round(durs[0], 2) if durs else 0.0,
        "max_sek": round(durs[-1], 2) if durs else 0.0,
    }

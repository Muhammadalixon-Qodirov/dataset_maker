"""O'tkan kunlar: 10 daqiqalik audioga mos matnni PDF'dan ajratish.

Muammo: audio trek 1 = kitobning boshi, lekin aniq QAYERDAN QAYERGACHA — noma'lum.
Yechim: Kotib audioning birinchi va oxirgi 30 sekundini eshitadi, shu so'zlarni
PDF matnida qidiramiz (fuzzy), oraliqni kesib olamiz.

Chiqish (tests/otkan_kunlar/):
    text_10min.txt   — audioga mos deb topilgan matn (asl orfografiya, punktuatsiya bilan)
    namuna.md        — foydalanuvchi tekshirishi uchun: boshi/oxiri, Kotib eshitgani vs matn
"""

from __future__ import annotations

import io
import os
import re
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "D:/hf_cache")

D = os.path.join(HERE, "otkan_kunlar")
PDF = os.path.join(D, "otkan_kunlar.pdf")
WAV = os.path.join(D, "audio_10min.wav")

from src.quality import KotibASR, cer
from src.uz_text import romanize_for_align


def log(m: str) -> None:
    print(m, flush=True)


# ------------------------------------------------------------- 1. PDF matni
def extract_pdf(path: str, max_pages: int = 40) -> str:
    try:
        import fitz

        doc = fitz.open(path)
        pages = [doc[i].get_text() for i in range(min(len(doc), max_pages))]
        lib = "PyMuPDF"
    except ImportError:
        import PyPDF2

        r = PyPDF2.PdfReader(path)
        pages = [(r.pages[i].extract_text() or "") for i in range(min(len(r.pages), max_pages))]
        lib = "PyPDF2"
    log(f"[1] PDF: {lib}, {len(pages)} bet o'qildi")
    return "\n".join(pages)


def clean_pdf_text(t: str) -> str:
    # satr oxiridagi defisli bo'linish: "tari-\nximizning" -> "tariximizning"
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    # oddiy satr tugashi -> bo'shliq
    t = re.sub(r"\s*\n\s*", " ", t)
    # ziyouz PDF'da bet raqami/sarlavha takrorlanadi
    t = re.sub(r"www\.ziyouz\.com kutubxonasi\s*\d*", " ", t)
    t = re.sub(r"O[’'‘]tkan kunlar \(roman\)\. Abdulla Qodiriy", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# -------------------------------------------------- 2. fuzzy anchor topish
def find_anchor(text_words: list[str], query: str, search_from: int = 0,
                search_to: int | None = None) -> tuple[int, float]:
    """`query` (Kotib eshitgani) matn so'zlari ichida eng yaxshi mos keladigan
    joyni topadi. Qaytaradi: (so'z indeksi, CER)."""
    q = romanize_for_align(query)
    qn = len(q.split())
    if qn == 0:
        return -1, 1.0
    rom = [romanize_for_align(w) for w in text_words]
    best_i, best_c = -1, 1.0
    hi = len(text_words) - qn if search_to is None else min(search_to, len(text_words) - qn)
    for i in range(max(0, search_from), max(search_from, hi) + 1):
        window = " ".join(rom[i : i + qn])
        c = cer(q, window)
        if c < best_c:
            best_i, best_c = i, c
    return best_i, best_c


def main() -> int:
    raw = extract_pdf(PDF)
    text = clean_pdf_text(raw)
    words = text.split()
    log(f"[1] tozalangan matn: {len(words)} so'z (birinchi 40 bet)")

    # bob boshini topib, qidiruvni shu atrofdan boshlaymiz
    k = text.find("1264")
    start_hint = len(text[:k].split()) if k >= 0 else 0
    log(f"[1] '1264' (bob boshi) {start_hint}-so'zda")

    # ---------------------------------------------- 2. Kotib: boshi va oxiri
    wav, sr = sf.read(WAV, dtype="float32")
    dur = len(wav) / sr
    asr = KotibASR()
    head = asr.transcribe(wav[: int(30 * sr)])
    tail = asr.transcribe(wav[int((dur - 30) * sr) :])
    log(f"[2] Kotib BOSHI (0-30s):  \"{head[:160]}\"")
    log(f"[2] Kotib OXIRI (-30s):   \"{tail[:160]}\"")

    # ---------------------------------------------- 3. anchor
    # boshi: bob boshidan 300 so'z oldin/keyin oralig'ida (muqaddima bo'lishi mumkin)
    si, sc = find_anchor(words, head, max(0, start_hint - 400), start_hint + 400)
    # oxiri: boshidan keyin, 10 daq ≈ 1000-2500 so'z oralig'ida
    ei, ec = find_anchor(words, tail, si + 600, si + 3500)
    tail_n = len(romanize_for_align(tail).split())
    end_idx = ei + tail_n if ei >= 0 else -1
    log(f"[3] BOSHI anchor: so'z #{si}, CER {sc*100:.0f}%")
    log(f"[3] OXIRI anchor: so'z #{ei}..{end_idx}, CER {ec*100:.0f}%")
    if si < 0 or ei < 0 or end_idx <= si:
        log("    ❌ anchor topilmadi — matn qo'lda kesilishi kerak")
        return 1

    piece = " ".join(words[si:end_idx])
    io.open(os.path.join(D, "text_10min.txt"), "w", encoding="utf-8").write(piece)
    n = len(piece.split())
    log(f"[4] text_10min.txt: {n} so'z ({n/(dur/60):.0f} so'z/daqiqa)")

    # ---------------------------------------------- 4. namuna.md
    md = [
        "# O'tkan kunlar — sinov namunasi (ko'rib chiqish uchun)", "",
        "- Audio: `audio_10min.wav` — trek 1 ning birinchi **10.0 daqiqasi** (audiobook.uz, Malika Ibrohimova)",
        f"- Matn: `text_10min.txt` — **{n} so'z**, PDF (ziyouz.com) dan, Kotib bilan bog'langan",
        f"- Tezlik: ~{n/(dur/60):.0f} so'z/daqiqa",
        "", "## Boshi — mos keladimi?", "",
        f"**Kotib eshitdi (0–30 s):** {head}", "",
        f"**Matn boshi:** {' '.join(words[si:si+45])} …", "",
        f"_Moslik CER: {sc*100:.0f}%_", "",
        "## Oxiri — mos keladimi?", "",
        f"**Kotib eshitdi (oxirgi 30 s):** {tail}", "",
        f"**Matn oxiri:** … {' '.join(words[max(si, end_idx-45):end_idx])}", "",
        f"_Moslik CER: {ec*100:.0f}%_", "",
        "## Eslatma", "",
        "- Matn **asl orfografiyada** (Qodiriy tili: «qurilg'an», «botqan», «ila») — tizim uni o'zgartirmaydi",
        "- PDF'dan ekstrakt — satr bo'linishlari tuzatilgan, lekin ba'zi so'z ichidagi bo'shliqlar qolgan bo'lishi mumkin",
        "- 2 ta tekshiruv nuqtasi: audioni 0:00 va 9:30 da eshitib, yuqoridagi matn bilan solishtiring",
    ]
    io.open(os.path.join(D, "namuna.md"), "w", encoding="utf-8").write("\n".join(md))
    log(f"[4] namuna.md yozildi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

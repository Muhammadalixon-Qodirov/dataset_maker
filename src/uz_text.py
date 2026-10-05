"""O'zbek matn qatlami — TTS dataset uchun matn normalizatsiyasi.

Ikki chiqish beradi:
  normalize_for_dataset(t) -> datasetga yoziladigan toza o'zbek matni
  romanize_for_align(t)    -> MMS_FA alignment uchun sodda a-z + ' matni

Mavjud kutubxonalardan foydalanadi:
  uztext      — apostrof kanonlashtirish, kirill<->lotin, 2026 alifbo, pul (sum_words)
  uztextnorm  — sana, foiz, kasr, rim raqamlar, butun sonlar

O'zi hal qiladigan (kutubxonalarda yo'q yoki buzuq):
  - telefon raqami    -> raqam-raqam o'qish
  - vaqt HH:MM        -> so'z
  - pul               -> uztext.sum_words (uztextnorm bu yerda XATO qiladi)
  - qisqartmalar      -> kengaytirish
  - yakuniy apostrof  -> uztextnorm ASCII ' qoldiradi, qayta kanonlashtiriladi
"""

from __future__ import annotations

import re
import unicodedata

from uztext import normalize as _uz_normalize
from uztext import sum_words as _sum_words
from uztextnorm import UzbekTextNormalizer
from uztextnorm.dates import int_to_uzbek

_normalizer = UzbekTextNormalizer()

# Kanonik apostroflar (uztext ishlatadigan)
TURNED_COMMA = "ʻ"  # oʻ / gʻ uchun
MOD_APOS = "ʼ"  # tutuq belgisi

# Apostrofning barcha ko'rinishlari — romanizatsiya uchun
_APOS_VARIANTS = "'ʻʼ‘’‛`´ʹˊˋ＇"

# Qisqartmalar. Faqat ma'nosi bir xil bo'lganlar — noaniqlari ataylab yo'q
# (masalan "g." yil/gramm, "s." soat/sahifa — kontekstsiz hal qilinmaydi).
ABBREVIATIONS = {
    "prof": "professor",
    "dots": "dotsent",
    "akad": "akademik",
    "t.f.d": "tibbiyot fanlari doktori",
    "f.f.d": "falsafa fanlari doktori",
    "i.f.d": "iqtisod fanlari doktori",
    "va h.k": "va hokazo",
    "h.k": "hokazo",
    "va b": "va boshqalar",
    "v.b": "va boshqalar",
    "shu k": "shu kabi",
    "m-n": "masalan",
    "masl": "masalan",
}

# Birlik belgilari
UNIT_WORDS = {
    "km": "kilometr",
    "kg": "kilogramm",
    "mm": "millimetr",
    "cm": "santimetr",
    "ml": "millilitr",
    "kv": "kvadrat",
    "mln": "million",
    "mlrd": "milliard",
}

_DIGIT_WORDS = {
    "0": "nol",
    "1": "bir",
    "2": "ikki",
    "3": "uch",
    "4": "to'rt",
    "5": "besh",
    "6": "olti",
    "7": "yetti",
    "8": "sakkiz",
    "9": "to'qqiz",
}


def _digits_to_words(digits: str) -> str:
    """Raqamlarni bittalab o'qish: '998' -> 'to'qqiz to'qqiz sakkiz'."""
    return " ".join(_DIGIT_WORDS[d] for d in digits if d in _DIGIT_WORDS)


# ---------------------------------------------------------------- telefon

# +998 90 123 45 67 | 998901234567 | (90) 123-45-67 | 90 123 45 67
_PHONE_RE = re.compile(
    r"(?<![\d])"
    r"(\+?998[\s\-().]*)?"
    r"(\(?\d{2}\)?[\s\-.]*)"
    r"(\d{3}[\s\-.]*\d{2}[\s\-.]*\d{2})"
    r"(?![\d])"
)


def _phone_repl(m: re.Match) -> str:
    parts = []
    if m.group(1):
        parts.append("plus " + _digits_to_words(re.sub(r"\D", "", m.group(1))))
    for g in (m.group(2), m.group(3)):
        parts.append(_digits_to_words(re.sub(r"\D", "", g)))
    return " ".join(p for p in parts if p)


def expand_phones(text: str) -> str:
    return _PHONE_RE.sub(_phone_repl, text)


# ------------------------------------------------------------------- vaqt

_TIME_RE = re.compile(r"(?<![\d:])([01]?\d|2[0-3])[:.]([0-5]\d)(?![\d:])")


def _time_repl(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    out = int_to_uzbek(hour)
    if minute == 0:
        return out
    if minute < 10:
        return f"{out} nol {int_to_uzbek(minute)}"
    return f"{out} {int_to_uzbek(minute)}"


def expand_times(text: str) -> str:
    return _TIME_RE.sub(_time_repl, text)


# -------------------------------------------------------------------- pul

# uztextnorm pul summalarini XATO o'qiydi ("1 250 000,50" -> 125 million).
# uztext.sum_words to'g'ri ishlaydi va natijaga "soʻm"/"tiyin" ni o'zi qo'shadi.
_NUM_AMOUNT = r"(\d(?:[\d\s ]*\d)?(?:[,.]\d{1,2})?)"
_SOM_RE = re.compile(
    _NUM_AMOUNT + r"\s*so[" + re.escape(_APOS_VARIANTS) + r"]?m\b",
    re.IGNORECASE,
)
_USD_RE = re.compile(r"\$\s*" + _NUM_AMOUNT)


def _som_repl(m: re.Match) -> str:
    try:
        return _sum_words(m.group(1).strip())
    except Exception:
        return m.group(0)


def _usd_repl(m: re.Match) -> str:
    raw = re.sub(r"[\s ]", "", m.group(1)).replace(",", ".")
    try:
        if "." in raw:
            whole, frac = raw.split(".", 1)
            out = f"{int_to_uzbek(int(whole))} dollar {int_to_uzbek(int(frac))} sent"
        else:
            out = f"{int_to_uzbek(int(raw))} dollar"
    except Exception:
        return m.group(0)
    return f" {out} "  # yon so'zga yopishib qolmasin; keyin bo'shliqlar yig'iladi


def expand_money(text: str) -> str:
    text = _SOM_RE.sub(_som_repl, text)
    return _USD_RE.sub(_usd_repl, text)


# ------------------------------------------------------------ qisqartmalar


def expand_abbreviations(text: str) -> str:
    # Uzunini birinchi almashtirish ("va h.k" dan oldin "h.k" bo'lmasin)
    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        pattern = re.escape(abbr).replace(r"\ ", r"\s+").replace(r"\.", r"\s*\.\s*")
        text = re.sub(
            rf"(?<![\w]){pattern}\s*\.", ABBREVIATIONS[abbr], text, flags=re.IGNORECASE
        )
    for unit, word in UNIT_WORDS.items():
        text = re.sub(rf"(?<=\d)\s*{unit}\b", f" {word}", text, flags=re.IGNORECASE)
    text = text.replace("№", " raqam ").replace("&", " va ")
    return text


# ------------------------------------------------ rim raqami himoyasi

_ROMAN_ONLY = re.compile(r"^[IVXLCDM]+$")


def guard_roman(text: str) -> str:
    """Katta harfli so'zlarni kichraytiradi — faqat HAQIQIY rim raqamlari qoladi.

    Nega: uztextnorm katta harfli so'z ichidagi rim harflarini raqam deb o'qiydi.
    O'tkan kunlar sinovida (2026-10-02) bob sarlavhasi "OTABEK YUSUFBEK HOJI O‘G‘LI"
    dagi "LI" -> 51 -> "o'g'ellik birinchi" bo'lib chiqdi. Kichik harfda to'g'ri.

    Haqiqiy rim raqami (XXI, IV, LI) faqat I V X L C D M harflaridan iborat —
    ular tegilmaydi. Qolgan katta harfli so'zlar (MALIKA, O‘G‘LI) kichraytiriladi;
    uztextnorm baribir hammasini kichik qiladi, shuning uchun yo'qotish yo'q.
    """

    def fix(m: re.Match) -> str:
        w = m.group(0)
        core = re.sub(r"[^A-Za-z]", "", w)
        if len(core) >= 2 and core.isupper() and not _ROMAN_ONLY.match(core):
            return w.lower()
        return w

    return re.sub(r"\S+", fix, text)


# ------------------------------------------------------- asosiy funksiyalar


def normalize_for_dataset(text: str) -> str:
    """Datasetga yoziladigan matn: raqamlar so'zga, apostroflar kanonik."""
    text = unicodedata.normalize("NFC", text)
    text = _uz_normalize(text)  # apostroflarni kanonlashtirish
    text = expand_phones(text)  # umumiy sonlardan OLDIN
    text = expand_times(text)  # umumiy sonlardan OLDIN
    text = expand_money(text)  # uztextnorm'ning xatosini chetlab o'tish
    text = expand_abbreviations(text)
    text = guard_roman(text)  # uztextnorm'dan OLDIN — "O‘G‘LI" -> 51 xatosi
    text = _normalizer.normalize(text)  # sana, foiz, kasr, rim, butun sonlar
    text = _uz_normalize(text)  # uztextnorm ASCII ' qoldiradi — qayta tuzatish
    return re.sub(r"\s+", " ", text).strip()


_ALIGN_KEEP = re.compile(r"[^a-z' ]")


def romanize_for_align(text: str) -> str:
    """MMS_FA uchun: kichik harf, faqat a-z + ' + bo'shliq.

    MMS_FA vocab'i 29 token: a-z, ', *, -. U+02BB (ʻ) yo'q, shuning uchun
    apostrofning hamma ko'rinishi ASCII ' ga xaritalanadi.
    """
    text = unicodedata.normalize("NFC", text).lower()
    for ch in _APOS_VARIANTS:
        text = text.replace(ch, "'")
    # Lotin bo'lmagan diakritiklarni yechish (ö -> o, ş -> s — 2026 alifbosi)
    text = text.replace("ö", "o").replace("ğ", "g'").replace("ş", "sh").replace("ç", "ch")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _ALIGN_KEEP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare(text: str) -> tuple[str, str]:
    """Bitta qo'ng'iroqda ikkisini ham qaytaradi: (dataset matni, align matni)."""
    ds = normalize_for_dataset(text)
    return ds, romanize_for_align(ds)


def split_aligned(text: str) -> list[tuple[str, str]]:
    """Matnni so'zlarga ajratadi va har biriga align shaklini juftlaydi.

    Qaytaradi: [(dataset_so'zi, align_so'zi), ...] — INDEKSLARI 1:1 mos.

    Nega kerak: dataset matni punktuatsiya bilan ("gapning." ), align matni esa
    faqat a-z + '. Butun satrni romanizatsiya qilsa so'zlar soni mos kelmay
    qolishi mumkin; shuning uchun har so'z alohida o'giriladi va align shakli
    bo'sh chiqqanlari (masalan yolg'iz "—") tashlab yuboriladi.
    """
    out: list[tuple[str, str]] = []
    for token in normalize_for_dataset(text).split():
        al = romanize_for_align(token)
        if not al:
            continue
        # bitta so'z romanizatsiyadan keyin ikkiga bo'linib ketmasin
        out.append((token, al.replace(" ", "")))
    return out


SENTENCE_END = (".", "?", "!", "…", ":", ";")


def ends_sentence(ds_word: str) -> bool:
    """Dataset so'zi gap oxirini bildiradimi (tirnoq/qavs ham hisobga olinadi)."""
    stripped = ds_word.rstrip("\"'»)]”’»")
    return stripped.endswith(SENTENCE_END)


if __name__ == "__main__":
    tests = [
        "Narxi 1 250 000,50 so'm.",
        "2026-yil 10-sentyabrda qabul qilindi.",
        "Aholining 15% i, ya'ni 3/4 qismi.",
        "XXI asr boshida.",
        "Telefon: +998 90 123 45 67",
        "prof. A. Qodirov va h.k.",
        "Soat 14:30 da uchrashamiz.",
        "Masofа 25 km, og'irligi 3 kg.",
        "Narxi $150 edi.",
        "O`zbekiston san'at g'alaba",
    ]
    for t in tests:
        ds, al = prepare(t)
        print("IN :", t)
        print("DS :", ds)
        print("AL :", al)
        print()

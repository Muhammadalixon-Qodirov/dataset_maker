"""English transcript -> (display word, align word) pairs for MMS_FA.

MMS_FA's vocabulary is lowercase a-z plus the apostrophe, so numbers are spelled out
and everything else is dropped. The display word keeps the original punctuation.
"""

from __future__ import annotations

import re

from num2words import num2words

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_ORD = re.compile(r"^(\d+)(st|nd|rd|th)$", re.I)


def _spell(token: str) -> str:
    m = _ORD.match(token)
    if m:
        return num2words(int(m.group(1)), to="ordinal")
    return _NUM.sub(lambda x: " " + num2words(float(x.group().replace(",", ""))
                                              if "." in x.group() else int(x.group().replace(",", ""))) + " ",
                    token)


def align_form(word: str) -> str:
    w = _spell(word.strip()).lower().replace("’", "'").replace("‘", "'")
    w = w.replace("-", " ")
    w = re.sub(r"[^a-z' ]", "", w)
    return re.sub(r"\s+", " ", w).strip(" '")


def split_aligned(text: str) -> list[tuple[str, str]]:
    """[(display, align)] — one entry per aligner token, so numbers may expand."""
    out: list[tuple[str, str]] = []
    for raw in text.split():
        al = align_form(raw)
        if not al:
            continue
        parts = al.split()
        if len(parts) == 1:
            out.append((raw, parts[0]))
        else:
            # "1962" -> nineteen sixty two: the display text rides on the first token
            out.append((raw, parts[0]))
            out.extend(("", p) for p in parts[1:])
    return out


_VOWELS = re.compile(r"[aeiouy]+")


def syllables(word: str) -> int:
    """Vowel-group estimate; good enough for a speech-rate ratio between two readings."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(_VOWELS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)

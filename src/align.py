"""Forced alignment — uzun audioni ma'lum matnga moslashtirish.

torchaudio.pipelines.MMS_FA ishlatiladi: Meta'ning 1,130 tilda o'qitilgan
forced aligner'i. Windows'da C++ build kerak emas (ctc-forced-aligner
Windows'da yiqiladi — link error 1120).

Uzun audio strategiyasi:
  1. Audio bo'laklab modeldan o'tkaziladi (VRAM cheklovi) — ustma-ust bilan
  2. Emissionlar birlashtiriladi (ular kichik: 50 freym/sek x 29 token)
  3. forced_align BUTUN emission + BUTUN matn ustida bir marta ishlaydi

28 daqiqali audio uchun emission ~10 MB — xotirada muammo yo'q.
Audio forward pass esa bo'laklanadi, chunki 8 GB VRAM uzun audioni ko'tarmaydi.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as _BUNDLE

SAMPLE_RATE = _BUNDLE.sample_rate  # 16000
STAR = "*"


@dataclass
class WordSpan:
    """Bitta so'zning vaqt chegaralari va ishonch bahosi."""

    word: str
    start: float  # sekund
    end: float  # sekund
    score: float  # 0..1, CTC ishonchi

    @property
    def duration(self) -> float:
        return self.end - self.start


class ForcedAligner:
    def __init__(self, device: str | None = None, with_star: bool = True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.with_star = with_star
        self.model = _BUNDLE.get_model(with_star=with_star).to(self.device).eval()
        self.tokenizer = _BUNDLE.get_tokenizer()
        self.aligner = _BUNDLE.get_aligner()
        self._fps: float | None = None  # freym/sekund, birinchi o'tishda aniqlanadi

    # ------------------------------------------------------------- audio

    @staticmethod
    def load_audio(path: str) -> torch.Tensor:
        """16 kHz mono tensor qaytaradi, shape (samples,)."""
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        return wav.squeeze(0)

    # --------------------------------------------------------- emissions

    @torch.inference_mode()
    def emissions(
        self, wav: torch.Tensor, chunk_sec: float = 20.0, overlap_sec: float = 2.0
    ) -> torch.Tensor:
        """Bo'laklab emission hisoblaydi, ustma-ust qismlarni kesib birlashtiradi.

        Qaytaradi: (time, tokens) log-probability tensori (CPU'da).
        """
        chunk = int(chunk_sec * SAMPLE_RATE)
        overlap = int(overlap_sec * SAMPLE_RATE)
        step = chunk - overlap
        if step <= 0:
            raise ValueError("overlap_sec chunk_sec dan kichik bo'lishi kerak")

        n = wav.numel()
        if n <= chunk:
            emis, _ = self.model(wav.unsqueeze(0).to(self.device))
            out = emis[0].cpu()
            self._fps = out.shape[0] / (n / SAMPLE_RATE)
            return out

        pieces: list[torch.Tensor] = []
        for start in range(0, n, step):
            end = min(start + chunk, n)
            seg = wav[start:end]
            if seg.numel() < SAMPLE_RATE // 10:  # 0.1 sekunddan qisqa qoldiq
                break
            emis, _ = self.model(seg.unsqueeze(0).to(self.device))
            e = emis[0].cpu()

            # freym/sample nisbati — faqat bir marta o'lchanadi
            ratio = e.shape[0] / seg.numel()
            drop_l = 0 if start == 0 else int(overlap / 2 * ratio)
            drop_r = 0 if end >= n else int(overlap / 2 * ratio)
            pieces.append(e[drop_l : e.shape[0] - drop_r])

            if end >= n:
                break

        out = torch.cat(pieces, dim=0)
        self._fps = out.shape[0] / (n / SAMPLE_RATE)
        return out

    # --------------------------------------------------------- alignment

    def align(self, emission: torch.Tensor, words: list[str]) -> list[WordSpan]:
        """Emission + so'zlar ro'yxati -> so'z darajasidagi WordSpan'lar.

        `words` — romanize_for_align() dan o'tgan, faqat a-z + ' bo'lgan so'zlar.
        """
        if not words:
            return []
        if self._fps is None:
            raise RuntimeError("avval emissions() ni chaqiring")

        tokens = self.tokenizer(words)
        spans = self.aligner(emission, tokens)

        out: list[WordSpan] = []
        for word, span_seq in zip(words, spans):
            if not span_seq:
                continue
            start = span_seq[0].start / self._fps
            end = span_seq[-1].end / self._fps
            # TokenSpan.score allaqachon normallashgan ehtimollik
            score = sum(s.score for s in span_seq) / len(span_seq)
            out.append(WordSpan(word, start, end, float(score)))
        return out

    # ------------------------------------------------- uzun audio (bloklab)

    def align_long(
        self,
        wav: torch.Tensor,
        words: list[str],
        sent_end: list[bool],
        block_sec: float = 300.0,
        max_wps: float = 2.5,
        log=None,
    ) -> list[WordSpan]:
        """Soatlab audioni ketma-ket bloklar bilan moslashtiradi.

        Nega kerak: forced_align xotirasi (audio freymlari x matn tokenlari) ga
        ko'payadi. 10 daqiqa sig'adi, 2 soat sig'maydi. Shuning uchun:

          1. Audiodan `block_sec` lik blok olinadi.
          2. Matndan shu blokka ORTIG'I bilan so'z olinadi (tezlik noma'lum).
          3. Moslashtiriladi. Ortiqcha so'zlar blok oxiriga siqiladi va past ball
             oladi — ball tushgan joy topiladi.
          4. Undan oldingi oxirgi gap chegarasigacha qabul qilinadi; keyingi blok
             aynan shu joydan (audio va matnda) boshlanadi.

        Matn audiodan uzun bo'lishi mumkin (butun kitob) — audio tugaganda to'xtaydi.
        Qaytadi: moslashtirilgan so'zlar (words[:len(natija)] ga mos).
        """
        n = wav.numel()
        dur = n / SAMPLE_RATE
        out: list[WordSpan] = []
        t0, w0, rate = 0.0, 0, None
        K = 15  # ball o'rtachasi oynasi (so'z)

        while t0 < dur - 2.0 and w0 < len(words):
            t1 = min(dur, t0 + block_sec)
            last = t1 >= dur
            blk_dur = t1 - t0
            wps = max_wps if rate is None else min(max_wps, rate * 1.3)
            cand = words[w0 : w0 + int(blk_dur * wps) + 20]

            emis = self.emissions(wav[int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)])
            spans = self.align(emis, [STAR, *cand])
            spans = [s for s in spans if s.word != STAR]
            if len(spans) != len(cand):
                raise RuntimeError(f"blok {t0:.0f}s: span soni mos emas "
                                   f"({len(spans)} != {len(cand)})")

            # ball hali yaxshi bo'lgan oxirgi so'z (orqadan qidiriladi)
            guard = 0.0 if last else 5.0
            good = -1
            for i in range(len(spans) - 1, K - 2, -1):
                window = spans[i - K + 1 : i + 1]
                if (sum(s.score for s in window) / K >= 0.35
                        and spans[i].end <= blk_dur - guard):
                    good = i
                    break
            if good < 0:
                if log:
                    log(f"      ⚠️ blok {t0/60:.1f} daq: ishonchli moslik topilmadi — to'xtadi")
                break

            j = good
            if not last:
                # gap chegarasiga qaytamiz, xavfsizlik uchun yana bitta gap orqaga
                ends = [i for i in range(good + 1) if sent_end[w0 + i]]
                if len(ends) >= 2:
                    j = ends[-2]
                elif ends:
                    j = ends[-1]
            if j < 5 and not last:
                if log:
                    log(f"      ⚠️ blok {t0/60:.1f} daq: oldinga siljish yo'q — to'xtadi")
                break

            for s in spans[: j + 1]:
                out.append(WordSpan(s.word, s.start + t0, s.end + t0, s.score))

            end_t = spans[j].end
            if j + 1 < len(spans) and spans[j + 1].start > end_t:
                end_t = (end_t + spans[j + 1].start) / 2
            rate = (j + 1) / max(end_t, 1.0)
            if log:
                log(f"      blok {t0/60:6.1f}–{t1/60:6.1f} daq: {j+1} so'z "
                    f"({rate*60:.0f} so'z/daq), jami {len(out)}")
            w0 += j + 1
            t0 += end_t
            if last:
                break

        return out

    # ------------------------------------------------------------ qulaylik

    def align_file(
        self,
        audio_path: str,
        words: list[str],
        chunk_sec: float = 20.0,
        overlap_sec: float = 2.0,
    ) -> tuple[list[WordSpan], float]:
        """Fayl + so'zlar -> (WordSpan'lar, audio davomiyligi sekundda)."""
        wav = self.load_audio(audio_path)
        emission = self.emissions(wav, chunk_sec, overlap_sec)
        return self.align(emission, words), wav.numel() / SAMPLE_RATE


def add_stars(words: list[str], mode: str = "edges") -> list[str]:
    """Wildcard `*` token qo'shish — o'quvchi matndan chetga chiqsa yordam beradi.

    'none'    — qo'shilmaydi (matn audioga aniq mos kelsa)
    'edges'   — faqat boshi va oxiriga (audio boshi/oxirida ortiqcha narsa bo'lsa)
    'between' — har so'z orasiga (o'quvchi so'z qo'shib/tashlab ketsa)
    """
    if mode == "none":
        return list(words)
    if mode == "edges":
        return [STAR, *words, STAR]
    if mode == "between":
        out = [STAR]
        for w in words:
            out.extend([w, STAR])
        return out
    raise ValueError(f"noma'lum mode: {mode}")


def strip_stars(spans: list[WordSpan]) -> list[WordSpan]:
    """Natijadan `*` span'larini olib tashlash."""
    return [s for s in spans if s.word != STAR]

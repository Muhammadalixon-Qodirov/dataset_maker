"""Sifat filtri — segmentlarni avtomatik baholash.

Eng kuchli filtr: ASR back-check. Har klipni QAYTA transkripsiya qilib,
tayinlangan matnga solishtiramiz. Mos kelmasa — alignment xato qilgan.

⚠️ O'zbek uchun Whisper YARAMAYDI (WER 90.2%). Kotib ishlatiladi (CER 7.0%).

Chegaralar — manbasi ko'rsatilgan:
  CER <= 5%            qabul            [IndicVoices-R]
  CER 5-15%            qo'lda ko'rish   [IndicVoices-R]
  CER > 15%            rad              [IndicVoices-R]
  edge CER <= 60%      chetlar yumshoq  [NeMo CER_EDGE_THRESHOLD=60, EDGE_LEN=5]
  SNR >= 20 dB         qabul            [LibriTTS "clean" mezoni]
  min_score >= 0.25    alignment ishonchi  [BIZNING tanlov — Kürzinger 0.22 ni
                       eslatadi, lekin u ctc-segmentation shkalasida. MMS_FA
                       score'i 0..1 ehtimollik, shkala boshqa. Kalibrlash kerak.]
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# --- chegaralar (manbalar yuqorida) ---
# ⚠️ CER chegarasi ASR'ning O'Z xatosidan YUQORI bo'lishi SHART.
# Kotib CER ~7.0%. FLEURS uz_uz ground-truth matni bilan o'lchandi (2026-10-02):
#   qabul qilinganlar CER median 0.00% | rad etilganlar median 7.9%
# 5% chegara bilan 9 ta YAXSHI klip faqat Kotib'ning o'z xatosi uchun belgilangan edi.
CER_ACCEPT = 0.10  # Kotib 7% floor + zaxira
CER_REVIEW = 0.20
EDGE_CER_MAX = 0.60
EDGE_LEN = 5
SNR_MIN = 20.0

# ⚠️ min_score NOTO'G'RI statistika: segmentdagi ENG PAST so'zni oladi, ya'ni
# bitta qisqa so'z ("u", "va") butun segmentni yiqitadi. O'lchovda qabul
# qilingan segmentlarda ham min 0.000 uchradi, mean esa 0.533.
# Shuning uchun asosiy darvoza — MEAN, min uchun faqat juda past pol.
MEAN_ALIGN_SCORE = 0.35  # bizning kalibrlangan qiymat (FLEURS o'lchovi)
MIN_ALIGN_FLOOR = 0.02  # bundan past = alignment butunlay yiqilgan

# Lokal nusxa bo'lsa — o'sha, bo'lmasa HF Hub'dan (bulut: HF Spaces / Colab).
# KOTIB_PATH muhit o'zgaruvchisi bilan almashtirish mumkin.
_KOTIB_LOCAL = os.path.expanduser("~/AI_studio/stt_models/kotib_uzbek_stt")
KOTIB_PATH = os.environ.get(
    "KOTIB_PATH", _KOTIB_LOCAL if os.path.isdir(_KOTIB_LOCAL) else "Kotib/uzbek_stt_v1"
)


# ----------------------------------------------------------------- CER


def edit_distance(a: str, b: str) -> int:
    """Levenshtein masofasi (o'rin tejovchi, bir qatorli DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate. Ikkisi ham oldin bir xil normallashtirilgan bo'lsin."""
    ref = reference.strip()
    if not ref:
        return 1.0
    return edit_distance(ref, hypothesis.strip()) / len(ref)


def edge_cer(reference: str, hypothesis: str, n: int = EDGE_LEN) -> tuple[float, float]:
    """Boshi va oxiridagi n belgi uchun alohida CER (chegara xatolarini ushlaydi)."""
    ref, hyp = reference.strip(), hypothesis.strip()
    if len(ref) < n or len(hyp) < n:
        return 0.0, 0.0
    return cer(ref[:n], hyp[:n]), cer(ref[-n:], hyp[-n:])


# ----------------------------------------------------------------- SNR


def estimate_snr(wav: np.ndarray, speech_mask: np.ndarray | None = None) -> float | None:
    """Nutq/jimlik energiyasi nisbati bo'yicha SNR (dB).

    `speech_mask` — har sample uchun True/False (VAD'dan). Berilmasa, energiya
    taqsimotining pastki 10% i shovqin deb olinadi (qo'pol, lekin halol baho).

    ⚠️ Bu WADA-SNR EMAS. WADA alohida amplitude taqsimot jadvalini talab qiladi;
    bu yerda soddaroq energiya nisbati ishlatilgan. Qiymatlar WADA bilan
    to'g'ridan-to'g'ri solishtirilmasligi kerak.
    """
    if wav.size == 0:
        return None
    if speech_mask is not None and speech_mask.any() and (~speech_mask).any():
        sp = wav[speech_mask]
        ns = wav[~speech_mask]
    else:
        frame = 400  # 25 ms @ 16 kHz
        n = wav.size // frame
        if n < 4:
            return None
        energies = (wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
        order = np.argsort(energies)
        noise_idx = order[: max(1, n // 10)]
        speech_idx = order[-max(1, n // 2) :]
        ns = np.concatenate([wav[i * frame : (i + 1) * frame] for i in noise_idx])
        sp = np.concatenate([wav[i * frame : (i + 1) * frame] for i in speech_idx])

    p_noise = float((ns**2).mean())
    p_speech = float((sp**2).mean())
    if p_noise <= 0 or p_speech <= 0:
        return None
    return 10.0 * np.log10(p_speech / p_noise)


# ------------------------------------------------------------ Kotib ASR


class KotibASR:
    """O'zbek ASR — segmentlarni qayta transkripsiya qilish uchun.

    Whisper o'zbekda 90.2% WER beradi, shuning uchun u YARAMAYDI.
    Kotib CER 7.0% — shuning uchun back-check filtri ishlashi mumkin.
    ⚠️ Kotib'ning o'z xatosi ham bor: CER chegarasini 2-3% dan pastga
    tushirish ma'nosiz (ParsVoice'da Whisper uchun ~2% floor topilgan).
    """

    # Whisper bir segment uchun ko'pi bilan 448 token chiqaradi. Chegarasiz qolsa,
    # notanish orfografiyada (Qodiriy 1926) EOS'ga yetguncha uzoq "o'ylaydi".
    # O'zbek lotini BPE'da ~3-4 token/sek nutq; 8 = ikki baravar zaxira.
    TOKENS_PER_SEC = 8
    MAX_WHISPER_TOKENS = 440  # 448 minus maxsus tokenlar

    def __init__(self, model_path: str = KOTIB_PATH, device: str | None = None):
        import torch
        from transformers import pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        # fp16: ~2x tez, VRAM yarmi. 1.5B parametrli Kotib fp32 da ~6 GB olardi.
        # CPU'da fp16 sekin/qo'llanmaydi — fp32 qoladi.
        on_gpu = device != -1 and device != "cpu" and torch.cuda.is_available()
        dtype = torch.float16 if on_gpu else torch.float32
        try:  # transformers 5.x: `dtype`
            self.pipe = pipeline("automatic-speech-recognition", model=model_path,
                                 device=device, dtype=dtype)
        except TypeError:  # transformers 4.x: `torch_dtype`
            self.pipe = pipeline("automatic-speech-recognition", model=model_path,
                                 device=device, torch_dtype=dtype)

    @classmethod
    def _max_tokens(cls, dur_sec: float) -> int:
        return int(min(cls.MAX_WHISPER_TOKENS, max(32, dur_sec * cls.TOKENS_PER_SEC + 16)))

    # 28 sekunddan qisqa audio uchun chunk_length_s BERILMAYDI: u Whisper'ning
    # eksperimental uzun-audio yo'lini majburan yoqadi va juda sekinlashtiradi
    # ("Using chunk_length_s is very experimental with seq2seq models").
    # Segmentlar 2-12 sek, ya'ni hech qachon kerak emas.
    CHUNK_THRESHOLD = 28.0

    def _kwargs(self, dur: float, batch_size: int) -> dict:
        kw: dict = {"generate_kwargs": {
            "language": "uz", "task": "transcribe",
            "max_new_tokens": self._max_tokens(dur),
        }}
        if dur > self.CHUNK_THRESHOLD:
            kw["chunk_length_s"] = 30
            kw["batch_size"] = batch_size
        return kw

    def transcribe(self, wav16: np.ndarray, sr: int = 16000) -> str:
        """16 kHz mono float32 array -> matn."""
        kw = self._kwargs(wav16.size / sr, 4)
        try:
            r = self.pipe(wav16, **kw)
        except Exception:
            kw.pop("generate_kwargs", None)
            r = self.pipe(wav16, **kw)
        return r["text"].strip()

    def _run(self, clips: list[np.ndarray], batch_size: int, sr: int = 16000) -> list[str]:
        longest = max(c.size for c in clips) / sr
        kw = {"generate_kwargs": {
            "language": "uz", "task": "transcribe",
            "max_new_tokens": self._max_tokens(longest),
        }}
        try:
            out = self.pipe(list(clips), batch_size=batch_size, **kw)
        except TypeError:
            out = self.pipe(list(clips), batch_size=batch_size)
        if isinstance(out, dict):
            out = [out]
        return [(o.get("text") or "").strip() for o in out]

    def transcribe_batch(
        self,
        clips: list[np.ndarray],
        batch_size: int = 4,
        sr: int = 16000,
        on_progress=None,
    ) -> list[str]:
        """Ko'p klipni batch bilan — bittalab chaqirishdan ancha tez.

        OOM'ga chidamli: xotira yetmasa batch'ni YARMIGA tushirib qayta urinadi,
        1 gacha. 8 GB kartada Kotib (2.9 GB) uchun batch 16 OOM beradi — sinovda
        "14.53 GiB is allocated by PyTorch" xatosi chiqdi.
        """
        if not clips:
            return []
        out: list[str] = []
        i = 0
        bs = max(1, batch_size)
        while i < len(clips):
            chunk = clips[i : i + bs]
            try:
                out.extend(self._run(chunk, bs, sr))
            except Exception as e:
                oom = "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"
                if oom and bs > 1:
                    self._free()
                    bs = max(1, bs // 2)
                    continue  # shu bo'lakni kichikroq batch bilan qayta urinamiz
                if oom:
                    # batch=1 da ham sig'masa — bu klipni tashlab ketamiz
                    self._free()
                    out.append("")
                    i += 1
                    continue
                raise
            i += len(chunk)
            if on_progress:
                on_progress(min(i, len(clips)), len(clips), bs)
        return out

    @staticmethod
    def _free() -> None:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------- verdikt


@dataclass
class Verdict:
    ok: bool
    action: str  # "qabul" | "ko'rish" | "rad"
    reason: str
    cer: float | None = None
    snr: float | None = None
    align_score: float | None = None
    asr_text: str | None = None


def judge(
    ref_text_align: str,
    asr_text_align: str,
    snr: float | None,
    min_align_score: float,
    mean_align_score: float | None = None,
) -> Verdict:
    """Bitta segment uchun qaror.

    Ikki matn ham `uz_text.romanize_for_align()` dan o'tgan bo'lishi SHART —
    aks holda CER punktuatsiya va apostrof farqidan shishib ketadi.

    Alignment darvozasi MEAN score bo'yicha (min emas — yuqoridagi izohga qarang).
    """
    c = cer(ref_text_align, asr_text_align)
    e_start, e_end = edge_cer(ref_text_align, asr_text_align)
    mean_s = min_align_score if mean_align_score is None else mean_align_score

    def v(ok: bool, action: str, reason: str) -> Verdict:
        return Verdict(ok, action, reason, c, snr, mean_s, asr_text_align)

    # CER — ENG KUCHLI dalil. ASR klipni aynan tayinlangan matn deb eshitsa,
    # klip to'g'ri: so'z kesilgan bo'lsa ASR uni eshitmas edi va CER ko'tarilardi.
    # Shuning uchun toza CER alignment ball darvozasini CHETLAB O'TADI.
    # (O'lchov: 9 ta klip CER=0 bo'lsa ham min_score poli bilan rad etilgan edi.)
    clean_cer = c <= CER_ACCEPT and max(e_start, e_end) <= EDGE_CER_MAX

    if not clean_cer:
        if min_align_score < MIN_ALIGN_FLOOR:
            return v(False, "rad", f"alignment yiqilgan (min {min_align_score:.3f})")
        if mean_s < MEAN_ALIGN_SCORE:
            return v(False, "rad", f"alignment ishonchi past (mean {mean_s:.2f})")
    if snr is not None and snr < SNR_MIN:
        return v(False, "rad", f"SNR {snr:.1f} dB < {SNR_MIN}")
    if c > CER_REVIEW:
        return v(False, "rad", f"CER {c*100:.1f}% > {CER_REVIEW*100:.0f}%")
    if max(e_start, e_end) > EDGE_CER_MAX:
        return v(False, "ko'rish", f"chegara CER {max(e_start,e_end)*100:.0f}%")
    if c > CER_ACCEPT:
        return v(False, "ko'rish", f"CER {c*100:.1f}% — qo'lda tekshirish")
    return v(True, "qabul", f"CER {c*100:.1f}%")

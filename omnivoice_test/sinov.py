"""OmniVoice zero-shot sinovi: 10 sek referens -> 30 sek sintez -> Kotib baho.

Ishlatish:
    python omnivoice_test/sinov.py mening_ovozim.wav

Qadamlar:
  1. Referensni 16 kHz mono WAV ga o'girish (har qanday formatdan)
  2. Kotib bilan referens matnini olish (--ref_text uchun)
  3. omnivoice-infer: referens ovozida matn_30s.txt ni o'qitish
  4. Natijani Kotib bilan tekshirish — CER (tushunarlilik)
  5. Hisobot: omnivoice_test/natija/
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

OUT = os.path.join(HERE, "natija")
os.makedirs(OUT, exist_ok=True)


def log(m: str) -> None:
    print(m, flush=True)


def to_wav16(src: str, dst: str) -> float:
    """Har qanday audioni 16 kHz mono WAV ga. Davomiylikni (sek) qaytaradi."""
    try:
        data, sr = sf.read(src, dtype="float32")
    except Exception:
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
        tmp = dst + ".tmp.wav"
        subprocess.run([ff, "-y", "-i", src, "-ar", "16000", "-ac", "1", tmp],
                       capture_output=True, check=True)
        data, sr = sf.read(tmp, dtype="float32")
        os.remove(tmp)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    # peak normalize — juda past yoki juda yuqori yozuvni tenglashtirish
    peak = float(np.abs(data).max()) or 1.0
    data = data * (0.9 / peak)
    sf.write(dst, data, 16000, subtype="PCM_16")
    return len(data) / 16000


def main(ref_path: str) -> int:
    from src.quality import KotibASR, cer
    from src.uz_text import romanize_for_align

    t_all = time.time()
    text = io.open(os.path.join(HERE, "matn_30s.txt"), encoding="utf-8").read().strip()

    # ------------------------------------------------ 1. referens tayyorlash
    ref16 = os.path.join(OUT, "referens_16k.wav")
    dur = to_wav16(ref_path, ref16)
    log(f"[1/4] referens: {dur:.1f} sek -> {ref16}")
    if dur < 3:
        log("      ⚠️ 3 sekunddan qisqa — natija yomon bo'lishi mumkin")
    if dur > 30:
        log("      ⚠️ 30 sekunddan uzun — model birinchi qismini ishlatishi mumkin")

    # ------------------------------------------------ 2. referens matni (Kotib)
    log("[2/4] Kotib referensni eshityapti...")
    asr = KotibASR()
    ref_wav, _ = sf.read(ref16, dtype="float32")
    ref_text = asr.transcribe(ref_wav)
    log(f"      referensda: \"{ref_text}\"")
    io.open(os.path.join(OUT, "referens_matni.txt"), "w", encoding="utf-8").write(ref_text)

    # Kotib va OmniVoice VRAM'da birga sig'masligi mumkin — bo'shatamiz
    del asr
    import torch

    torch.cuda.empty_cache()

    # ------------------------------------------------ 3. OmniVoice sintez
    out_wav = os.path.join(OUT, "sintez_30s.wav")
    cmd = [
        "omnivoice-infer",
        "--ref_audio", ref16,
        "--ref_text", ref_text,
        "--language", "uz",
        "--text", text,
        "--output", out_wav,
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    log("[3/4] OmniVoice sintez qilyapti (birinchi marta model yuklanadi)...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    el = time.time() - t0
    if r.returncode != 0 or not os.path.exists(out_wav):
        log(f"      ❌ XATO (exit {r.returncode}):")
        log(r.stderr[-1500:])
        return 1
    syn, ssr = sf.read(out_wav, dtype="float32")
    syn_dur = len(syn) / ssr
    log(f"      ✅ {syn_dur:.1f} sek audio, {el:.0f} s da (RTF {el/syn_dur:.2f})")

    # ------------------------------------------------ 4. Kotib bilan baho
    log("[4/4] Kotib sintezni eshityapti...")
    asr = KotibASR()
    if syn.ndim > 1:
        syn = syn.mean(axis=1)
    if ssr != 16000:
        import librosa

        syn16 = librosa.resample(syn, orig_sr=ssr, target_sr=16000)
    else:
        syn16 = syn
    heard = asr.transcribe(syn16)
    c = cer(romanize_for_align(text), romanize_for_align(heard))
    log(f"      Kotib eshitdi: \"{heard[:150]}\"")
    # Python 3.10: f-string ichida backslash bo'lmaydi \u2014 belgini tashqarida hisoblaymiz
    if c < 0.05:
        label = "a'lo"
    elif c < 0.10:
        label = "yaxshi"
    elif c < 0.20:
        label = "o'rtacha"
    else:
        label = "yomon"
    log(f"      TUSHUNARLILIK: CER = {c*100:.1f}%  ({label})")

    report = {
        "referens_sek": round(dur, 1),
        "referens_matni_kotib": ref_text,
        "sintez_sek": round(syn_dur, 1),
        "sintez_vaqti_sek": round(el, 1),
        "rtf": round(el / syn_dur, 3),
        "berilgan_matn": text,
        "kotib_eshitgani": heard,
        "cer": round(c, 4),
        "jami_vaqt_sek": round(time.time() - t_all, 1),
    }
    json.dump(report, io.open(os.path.join(OUT, "hisobot.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    log("")
    log("=" * 60)
    log(f"ESHITING:  {out_wav}")
    log(f"Hisobot:   {os.path.join(OUT, 'hisobot.json')}")
    log("=" * 60)
    log("Baholash sizda: ovoz SIZNIKIGA o'xshaydimi? Tabiiymi?")
    log("CER tushunarlilikni o'lchaydi, O'XSHASHLIKNI faqat siz baholaysiz.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ishlatish: python omnivoice_test/sinov.py <ovoz_fayli>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))

"""AI Dataset Maker — uzun o'zbek audio + to'liq matn -> TTS dataset.

Ishlatish:
    python main.py --audio ovoz.wav --text kitob.txt --out dataset/

    # matn yo'q bo'lsa, Kotib bilan yasash:
    python main.py --audio ovoz.wav --asr-text --out dataset/

Quvur:
    matn    -> uz_text   (o'zbek normalizatsiya: apostrof, raqam, sana, qisqartma)
    audio   -> align     (MMS_FA forced alignment, bo'laklab)
            -> segment   (2-12 sek, gap chegarasida, so'z o'rtasidan kesmaydi)
            -> quality   (Kotib bilan ASR back-check, CER filtri)
            -> export    (LJSpeech: wavs/ + metadata.csv)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import pipeline as P


def main() -> int:
    p = argparse.ArgumentParser(description="O'zbek TTS dataset tayyorlovchi")
    p.add_argument("--audio", required=True, help="uzun audio fayl (wav/mp3)")
    p.add_argument("--text", help="to'liq matn fayli (.txt, UTF-8)")
    p.add_argument("--asr-text", action="store_true",
                   help="matn fayli yo'q — Kotib bilan transkripsiya yasash")
    p.add_argument("--out", required=True, help="chiqish papkasi")
    p.add_argument("--sr", type=int, default=22050,
                   help="chiqish sample rate (22050=VITS, 24000=XTTS/StyleTTS2)")
    p.add_argument("--star", default="edges", choices=["none", "edges", "between"],
                   help="wildcard token: matn audioga qanchalik mos kelmaydi")
    p.add_argument("--chunk", type=float, default=20.0, help="alignment bo'lak uzunligi (sek)")
    p.add_argument("--block", type=float, default=300.0,
                   help="uzun audio uchun moslashtirish bloki, sek (audio bundan 1.5x uzun bo'lsa)")
    p.add_argument("--max-pause", type=float, default=0.93,
                   help="segment ichidagi ruxsat etilgan eng uzun jimlik, sek. "
                        "0.93=oddiy o'qish (GLOBE) · 1.5=badiiy/dramatik o'qish")
    p.add_argument("--no-vad", action="store_true",
                   help="silero-VAD bilan chegara moslashtirishni o'chirish (faqat pad)")
    p.add_argument("--batch", type=int, default=4,
                   help="ASR back-check batch hajmi (8 GB VRAM uchun 4; OOM bo'lsa o'zi tushadi)")
    p.add_argument("--no-asr-check", action="store_true",
                   help="ASR back-check filtrini o'tkazib yuborish (tezroq, sifatsizroq)")
    p.add_argument("--device", default=None, help="cuda yoki cpu")
    args = p.parse_args()

    if not args.text and not args.asr_text:
        p.error("--text yoki --asr-text kerak")

    text = None
    if args.text:
        with open(args.text, encoding="utf-8") as f:
            text = f.read()

    opt = P.Options(
        sr=args.sr, star=args.star, chunk=args.chunk, block=args.block,
        max_pause=args.max_pause, vad=not args.no_vad, asr_check=not args.no_asr_check,
        batch=args.batch, device=args.device,
    )
    try:
        P.run(args.audio, args.out, text=text, opt=opt)
    except ValueError as e:
        print(f"XATO: {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

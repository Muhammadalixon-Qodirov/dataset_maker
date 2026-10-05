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
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import align as A
from src import export as E
from src import quality as Q
from src import segment as S
from src import uz_text as T


def log(msg: str) -> None:
    print(msg, flush=True)


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
    os.makedirs(args.out, exist_ok=True)
    t_all = time.time()

    # ---------------------------------------------------------- 1. audio
    log("[1/6] audio yuklanmoqda...")
    wav = A.ForcedAligner.load_audio(args.audio)  # 16 kHz mono tensor
    wav_np = wav.numpy()
    dur = wav.numel() / A.SAMPLE_RATE
    log(f"      {dur/60:.1f} daqiqa ({dur:.0f} sek)")

    # ----------------------------------------------------------- 2. matn
    asr: Q.KotibASR | None = None
    if args.text:
        with open(args.text, encoding="utf-8") as f:
            raw_text = f.read()
        log(f"[2/6] matn: {len(raw_text)} belgi")
    else:
        log("[2/6] Kotib bilan transkripsiya yasalmoqda (uzoq ketishi mumkin)...")
        asr = Q.KotibASR(device=args.device)
        raw_text = asr.transcribe(wav_np)
        with open(os.path.join(args.out, "asr_text.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)
        log(f"      {len(raw_text)} belgi (asr_text.txt ga saqlandi)")

    pairs = T.split_aligned(raw_text)
    words = [al for _, al in pairs]
    log(f"      normalizatsiyadan keyin {len(words)} so'z")
    if not words:
        log("XATO: matndan bitta so'z ham chiqmadi")
        return 1

    # Oraliq natijalar keshi: ish yarmida to'xtasa (kompyuter o'chsa, GPU band bo'lsa)
    # qayta ishga tushirganda tayyor qismi takrorlanmaydi. 2 soatlik sinovda 951
    # segmentdan 880 tasi tekshirilgach kompyuter o'chib, hammasi yo'qolgan edi.
    st = os.stat(args.audio)
    key = hashlib.sha1(
        f"{st.st_size}|{int(st.st_mtime)}|{args.block}|{args.star}|{' '.join(words)}".encode("utf-8")
    ).hexdigest()[:12]
    cache_align = os.path.join(args.out, f"_kesh_align_{key}.json")
    cache_asr = os.path.join(args.out, f"_kesh_asr_{key}.jsonl")

    # ------------------------------------------------------ 3. alignment
    log(f"[3/6] MMS_FA alignment (star={args.star})...")
    t0 = time.time()
    aligner = None
    if os.path.exists(cache_align):
        with open(cache_align, encoding="utf-8") as f:
            spans = [A.WordSpan(*row) for row in json.load(f)]
        log(f"      keshdan olindi: {len(spans)} so'z ({os.path.basename(cache_align)})")
    else:
        aligner = A.ForcedAligner(device=args.device)
        if dur > args.block * 1.5:
            # uzun audio: bir yo'la moslashtirish xotiraga sig'maydi — bloklab
            log(f"      uzun audio — {args.block:.0f} sekundlik bloklar bilan")
            sent_end = [T.ends_sentence(ds) for ds, _ in pairs]
            spans = aligner.align_long(wav, words, sent_end, block_sec=args.block, log=log)
        else:
            emission = aligner.emissions(wav, chunk_sec=args.chunk)
            spans = A.strip_stars(aligner.align(emission, A.add_stars(words, args.star)))
            del emission
        with open(cache_align, "w", encoding="utf-8") as f:
            json.dump([[s.word, s.start, s.end, s.score] for s in spans], f)
    el = time.time() - t0
    log(f"      {len(spans)}/{len(words)} so'z moslashtirildi — {el:.0f}s (RTF {el/dur:.3f})")
    if len(spans) != len(pairs):
        log(f"      matnning {len(spans)} so'zi audioga to'g'ri keldi (qolgani audiodan tashqarida)")
        k = min(len(spans), len(pairs))
        spans, pairs = spans[:k], pairs[:k]

    # MMS_FA ni VRAM'dan bo'shatamiz — Kotib 2.9 GB, 8 GB kartada ikkisi sig'masligi
    # mumkin (sinovda 7.3/8 GB ga chiqdi va OOM chegarasida ishladi).
    del aligner
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    # ----------------------------------------------------- 4. segmentlash
    log("[4/6] segmentlash...")
    vad_regions: list[tuple[float, float]] = []
    if not args.no_vad:
        from src import vad as V

        vad_regions = V.speech_regions(wav_np)
        log(f"      silero-VAD: {len(vad_regions)} nutq oralig'i"
            if vad_regions else "      ⚠️ VAD ishlamadi — faqat pad bilan davom etamiz")
    segments = S.build_segments(pairs, spans, dur, vad_regions=vad_regions,
                                max_inner_silence=args.max_pause)
    summ = S.summary(segments)
    log(f"      {summ['jami_segment']} segment | qabul {summ['qabul']} | rad {summ['rad']}")
    log(f"      {summ['jami_soat']} soat | o'rtacha {summ['ortacha_sek']}s "
        f"| {summ['min_sek']}-{summ['max_sek']}s")
    if summ["rad_sabablari"]:
        log(f"      rad sabablari: {summ['rad_sabablari']}")

    # ------------------------------------------------- 5. ASR back-check
    if args.no_asr_check:
        log("[5/6] ASR back-check O'TKAZIB YUBORILDI (--no-asr-check)")
    else:
        live = [s for s in segments if s.reject is None]
        log(f"[5/6] ASR back-check — {len(live)} segment (Kotib)...")
        if asr is None:
            asr = Q.KotibASR(device=args.device)
        t0 = time.time()
        clips = []
        for seg in live:
            a = int(seg.start * A.SAMPLE_RATE)
            b = min(int(seg.end * A.SAMPLE_RATE), wav_np.size)
            clips.append(wav_np[a:b])

        # Kesh: har segment natijasi darhol diskka yoziladi (kalit = vaqt oralig'i)
        seg_key = [f"{s.start:.3f}-{s.end:.3f}" for s in live]
        done_asr: dict[str, str] = {}
        if os.path.exists(cache_asr):
            with open(cache_asr, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done_asr[r["k"]] = r["t"]
                    except (ValueError, KeyError):
                        pass  # yarim yozilgan oxirgi qator
            log(f"      keshda {sum(1 for k in seg_key if k in done_asr)} segment tayyor")
        todo = [i for i, k in enumerate(seg_key) if k not in done_asr]

        # batch bilan — bittalab chaqirish juda sekin edi (timeout bo'ldi).
        # batch_size OOM bo'lganda o'zi yarmiga tushadi (8 GB karta uchun).
        STEP = 32
        with open(cache_asr, "a", encoding="utf-8") as fc:
            for c0 in range(0, len(todo), STEP):
                part = todo[c0 : c0 + STEP]
                texts = asr.transcribe_batch([clips[i] for i in part], batch_size=args.batch)
                for i, txt in zip(part, texts):
                    done_asr[seg_key[i]] = txt
                    fc.write(json.dumps({"k": seg_key[i], "t": txt}, ensure_ascii=False) + "\n")
                fc.flush()
                log(f"      {len(live)-len(todo)+c0+len(part)}/{len(live)} ({time.time()-t0:.0f}s)")
        hyps = [done_asr[k] for k in seg_key]

        for seg, clip, hyp in zip(live, clips, hyps):
            seg.verdict = Q.judge(
                T.romanize_for_align(seg.text),
                T.romanize_for_align(hyp),
                Q.estimate_snr(clip),
                seg.min_score,
                seg.mean_score,
            )
        acc = sum(1 for s in live if s.verdict.ok)
        rev = sum(1 for s in live if s.verdict.action == "ko'rish")
        rej = sum(1 for s in live if s.verdict.action == "rad")
        log(f"      qabul {acc} | ko'rish {rev} | rad {rej}")

    # --------------------------------------------------------- 6. eksport
    log(f"[6/6] eksport -> {args.out} ({args.sr} Hz)...")
    res = E.export_ljspeech(
        wav_np, A.SAMPLE_RATE, segments, args.out, target_sr=args.sr,
        extra={"manba_audio": os.path.basename(args.audio),
               "audio_daqiqa": round(dur / 60, 2),
               "segment_statistikasi": summ},
    )
    log(f"      yozildi {res.written} klip | rad {res.rejected} | {res.total_hours:.3f} soat")
    log(f"\nTAYYOR ({time.time()-t_all:.0f}s): {args.out}")
    log("  metadata.csv · wavs/ · rejected.csv · report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

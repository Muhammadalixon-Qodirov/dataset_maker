# AI Dataset Maker

**Uzun o'zbek audio + to'liq matn → TTS o'qitish uchun dataset.**

```bash
python main.py --audio kitob.wav --text kitob.txt --out dataset/
```

Chiqish: LJSpeech formati (`wavs/` + `metadata.csv`) + rad etilganlar hisoboti.

---

## Qayerdan boshlash

| Men nimani bilmoqchiman | Qaysi faylni o'qiyman |
|---|---|
| **Tizim ishlaydimi, nima qilinmagan?** | [NATIJALAR.md](NATIJALAR.md) ← **shundan boshlang** |
| Dataset qanday tayyorlanadi, qanday chegaralar | [TADQIQOT.md](TADQIQOT.md) |
| Ovoz klonlash (10 sekunddan) | [ZEROSHOT.md](ZEROSHOT.md) |
| TTS model ichidan qanday ishlaydi | [MODEL_ICHKI_QISMI.md](MODEL_ICHKI_QISMI.md) |
| Nega bunday xizmat yo'q, kim qilgan | [XIZMATLAR.md](XIZMATLAR.md) |
| Batafsil manbalar (856 KB, 362+ havola) | [tadqiqot/](tadqiqot/) |

---

## Hozirgi holat (2026-10-05)

```
✅ Tizim qurildi — ~1,600 qator, 7 fayl
✅ HAQIQIY DATASET YASADI — FLEURS uz_uz, 10.11 daqiqa:
      51 klip · 0.109 soat · saqlanish 75%
      CER median 2.11% · 41% da CER = 0
✅ Alignment 977/977 so'z · RTF 0.027 · chunking siljishi yo'q
✅ Chegaralar FLEURS ground-truth bilan kalibrlandi (3 iteratsiya)
✅ silero-VAD chegara snap ulandi
✅ 2 SOATLIK uzluksiz o'qishda sinaldi (2026-10-05):
      916 klip · 1.57 soat · CER median 2.2% · siljish yo'q
✅ Blokli moslashtirish (uzun audio) va to'xtagan joydan davom etish qo'shildi
⚠️ Chegaralar ikki manbada sozlangan — boshqa diktor/mikrofonda qayta tekshirish kerak
⚠️ Bu dataset bilan model o'qitish hali sinalmagan
```

Saqlanish 75% — nashr etilganlar darajasida (Libriheavy 83%, EuroSpeech 65%,
WenetSpeech4TTS 56%, Emilia 29%).

Batafsil: [NATIJALAR.md](NATIJALAR.md)

---

## Tuzilish

```
main.py              CLI, 6 bosqichli quvur
src/uz_text.py       o'zbek matn qatlami — apostrof, raqam, sana, qisqartma
src/align.py         MMS_FA forced alignment (uzun audio bo'laklab)
src/segment.py       2–12 sek kesish, gap chegarasida, 4 filtr
src/quality.py       Kotib ASR back-check, CER, SNR
src/vad.py           silero-VAD chegara snap
src/export.py        LJSpeech eksport, loudness, pad
tests/               FLEURS sinov + baholash skriptlari (qayta ishlatiladi)
```

---

## O'rnatish

```bash
pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install uztextnorm pyloudnorm soxr imageio-ffmpeg silero-vad
pip install "git+https://github.com/abafaboy/uztext#subdirectory=python"
```

Windows'da konsol uchun: `set PYTHONIOENCODING=utf-8`

⚠️ `ctc-forced-aligner` ni o'rnatmang — Windows'da build bo'lmaydi.
O'rniga `torchaudio.pipelines.MMS_FA` ishlatilgan (bir xil MMS modeli).

---

## Muhim bilib turish kerak

**1. Haqiqiy matn bilan ishlatilsin.** `--asr-text` rejimi Kotib
gallyutsinatsiyasi sababli ishonchsiz.

**2. Manba audio toza va bitta ovozli bo'lishi kerak.** Dialog, baqiriq,
fon shovqini — rad etiladi (va to'g'ri qiladi).

**3. 1 soat toza dataset uchun ~2 soat yozib olish kerak.** Professional
korpuslar kirishning 40–60%ini tashlab yuboradi (LibriTTS, Hi-Fi TTS).

**4. O'zbekda Whisper yaroqsiz** (WER 90.2%). Kotib ishlatiladi (CER 7.0%).

---

## Keyingi qadam

Tizimni haqiqiy materialda sinash:
- **FeruzaSpeech** (21.57 soat, 1 spiker, qo'lda yozilgan matn, CC BY 4.0), yoki
- o'zingiz 10 daqiqa kitob o'qib yozish

Batafsil: [NATIJALAR.md](NATIJALAR.md) 9-bo'lim.

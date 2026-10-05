# AI Dataset Maker — to'liq tadqiqot

**Vazifa:** foydalanuvchi tizimga (a) **to'liq matn** va (b) o'sha matnni o'qigan odamning **uzun audio** yozuvini beradi → tizim o'zi TTS o'qitish uchun dataset tayyorlaydi (qisqa, moslashtirilgan kliplar + mos transkripsiya).

**Tadqiqot sanasi:** 2026-09-29
**Manbalar:** 252 GitHub, 169 arXiv, 62 HuggingFace, NVIDIA/PyTorch rasmiy hujjatlari, ISCA arxivi
**Batafsil:** `tadqiqot/` papkasida 5 ta fayl, 3447 qator — bu hujjat ularning sintezi

---

## 0. Qisqa xulosa

1. **G'oyangiz to'g'ri va sinalgan.** Bu aynan LibriSpeech (2015) qurilgan usul. Sohada nomi — *forced alignment* / *long-form audio alignment*.
2. **Texnologiya hal bo'lgan:** CTC forced alignment. Eng yaxshi vosita — `ctc-forced-aligner`, o'zbek uchun akustik model kerak emas.
3. **Bozorda bo'shliq bor.** Hech bir mavjud vosita "to'liq matn + uzun audio → LJSpeech" ni Windows'da yaxshi qilmaydi. Ekosistema ikkiga bo'lingan va ular uchrashmaydi — siz aynan shu oraliqni to'ldirasiz.
4. **Eng muhim raqam:** 1 soat toza dataset uchun ~2 soat yozib olish kerak. Professional korpuslar kirishning 60%ini tashlab yuboradi.
5. **Sizda katta ustunlik bor:** Kotib STT (o'zbek CER 7%) allaqachon ishlaydi. Whisper o'zbekda **90.2% WER** — yaramaydi, shuning uchun Kotib hal qiluvchi.
6. **Tayyor dataset topildi:** FeruzaSpeech — 59.61 soat, bitta toshkentlik ayol spiker, CC BY 4.0. Uning kitob qismi (21.57 soat) aynan sizga kerak hajmda.
7. **"0 dan o'qitish"** klassik model uchun realistik (20–25 soat toza audio), zamonaviy zero-shot model uchun emas (27,000–95,000 soat).
8. **8 GB VRAM** dataset tayyorlashga to'liq yetadi, o'qitishga — yo'q.
9. **Asosiy falsafa:** maqsad "alignment'ni mukammal qilish" emas, balki **"qaysi segmentga ishonmaslik kerakligini aniq ko'rsatish."** Bu 10 soatlik qo'l mehnatini 1–3 soatga tushiradi.

---

## 1. Sizning g'oyangiz sohada qanday ataladi

| Nom | Ma'nosi |
|---|---|
| **Forced alignment** | Ma'lum matnni audio bilan vaqt bo'yicha moslashtirish |
| **Long-form audio alignment** | Soatlab davom etgan audio uchun shuning kengaytmasi |
| **Read-speech corpus construction** | O'qilgan nutq korpusini qurish |
| **Audiobook alignment** | Audiokitob + kitob matnini sinxronlash |

Bu **ASR (Whisper)dan printsipial farq qiladi:** ASR matnni *taxmin qiladi*, alignment esa matnni *biladi* va faqat vaqtini topadi. Sizning qo'lingizda haqiqiy matn bor — bu eng katta boylik, va ko'p vositalar uni behuda tashlab yuboradi.

---

## 2. Boshqalar qanday qilyapti — haqiqiy korpuslar

### LibriSpeech (2015) — sizning g'oyangizning asl nusxasi

```
LibriVox API + Internet Archive + Gutenberg RDF  (fuzzy title matching)
        ↓
LIUM diarization — ko'p o'quvchili boblarni tashlash
        ↓
1-o'tish: Kaldi VoxForge triphone bMMI
        ↓
Smith-Waterman → "ishonch orollari" (12+ mos fonema)
        ↓
2-o'tish: fonema darajasidagi bigram LM + fMLLR
        ↓
Segmentlar: ≤30 daq → ≤35 s → ≥0.3 s sukunatlarda bo'lish
```

- Spikerlarga **25–30 daqiqa** cheklov qo'yilgan
- **~1200 soat moslashtirilgan → 1000 soat qoldi** (~17% tashlandi): yomon Gutenberg matni, o'quvchi qo'shib/tashlab ketgan joylar, disfluensiyalar, normalizatsiya va G2P xatolari
- `clean`/`other` bo'linishi **nisbiy** WER reytingi (median bo'yicha), mutlaq chegara yo'q

### Tashlab yuborish statistikasi — eng muhim jadval

| Korpus | Qabul qilish mezoni | Qancha yo'qoldi |
|---|---|---|
| LibriSpeech | Smith-Waterman ishonch orollari | ~17% |
| **LibriTTS** (TTS uchun!) | **edit distance = 0** + WADA SNR ≥ 20 dB | ~30% alignment + ~25% SNR → **~60% qoldi** |
| **Hi-Fi TTS** | **faqat WER = 0**, bandwidth ≥13 kHz, SNR ≥32/40 dB | 250 o'quvchi → **10 ta** |
| Libriheavy | suffix array + Levenshtein, 2–30 s, faqat `.?!` da bo'lish | — |
| Emilia-Pipe | ko'p bosqichli | kirishning **29–39%** saqlanadi* |

\* v1 maqola va jurnal versiyasi bu raqamda ziddiyatda.

**LibriTTS eng muhim namuna** — u LibriSpeech'dan aynan TTS uchun ajratilgan: 24 kHz, asl + normalizatsiyalangan matn, gap chegarasida bo'lish, **nol edit distance** qoidasi.

### Muvaffaqiyatsizlik keyslari (eng qimmatli qism)

| Loyiha | Nima bo'ldi | Sababi |
|---|---|---|
| **People's Speech** | 30,000 soatning **44%** insondan yomon WER | Bo'sh `CER < 50%` filtri + alignment chegara xatolari |
| **LibriTTS** | TTS uchun juda shovqinli: MOS **2.59–2.97** | Restoration qilinmagan (LibriTTS-R: **4.06–4.31**) |
| **Common Voice** (filtrsiz) | MOS **2.35** | `WV-MOS ≥ 4.0` filtri 2,015 soatni **86 soatga** tushirdi, MOS 3.69 ga chiqdi |
| **YODAS** (auto subset) | CER **32.3%** | Avtomatik transkripsiyaga ishonish |

👉 **Saboq:** bo'sh filtr = ko'p ma'lumot, lekin yaroqsiz. Qattiq filtr = kam ma'lumot, lekin ishlaydi. TTS uchun **qattiq filtr har doim yutadi.**

---

## 3. Texnologiya: forced alignment

### Usullarni taqqoslash (Kürzinger 2020, arXiv:2007.09127v2, TEDlium v2)

| Usul | Vosita | O'rtacha xato | std | ±0.5 s ichida |
|---|---|---|---|---|
| **CTC** | ctc-segmentation | **0.31 s** | 0.85 s | **88.8%** |
| HMM | Gentle | 0.41 s | — | — |
| DTW | aeneas | **9.01 s** ❌ | — | — |

**Xulosa:** aeneas 30 baravar yomon. Kaldi/HMM (MFA, Gentle) va DTW (aeneas) — eski avlod. Soha CTC'ga o'tgan.

Maqolada berilgan ishonch chegarasi: **0.22 dan past segmentlarni tashlash** (log fazoda −1.5).

### ✅ Tavsiya: `ctc-forced-aligner` (MahmoudAshraf97)

**Nega aynan bu:**

1. **O'zbek uchun akustik model kerak emas** — default model MMS-300m (1130+ til, 31K soat), belgi darajasidagi CTC. O'zbek lotini deyarli `a-z` + `'`.
2. **MFA bu vazifada ishlamaydi:** o'zbek uchun rasmiy model yo'q, o'zi o'qitish uchun esa **allaqachon kesilgan kliplar kerak** — tovuq-tuxum muammosi. MFA birinchi qadam bo'la olmaydi.
3. **Uzun audio ichida hal qilingan:** `--window_size 30 --context_size 2`. "Kamida 5× kam xotira" deb da'vo qiladi — solishtirish uchun, `torchaudio.functional.forced_align` qattiq `batch_size == 1` cheklovi bilan.
4. **Windows'da muammosiz:** `pip install` + ffmpeg, `--device cuda`.
5. **Matn mos kelmasligiga tolerant:** `--star_frequency edges|segment` — wildcard `<star>` token, o'quvchi so'z tashlab ketsa ham ishlaydi.

⚠️ **Litsenziya tuzog'i:** kod BSD-2, lekin **MMS vaznlari CC-BY-NC 4.0** — tijorat loyihasi uchun modelni almashtirish shart.

### Windows holati (muhim)

| Vosita | Windows |
|---|---|
| `ctc-forced-aligner` | ✅ pip + ffmpeg |
| MFA 3.x | 🟡 hujjatlarda faqat Docker; MFA 2.1.7 native deydi — **manbalar ziddiyatda** |
| NeMo NFA | 🟡 Linux/Docker yo'nalishi, Python 3.12+ |
| Gentle | ❌ native yo'q, faqat ingliz |
| aeneas | ❌ espeak linking xatolari, uzun kesishda xotira muammosi |
| `afaligner` | ❌ README'da "Windows qo'llanmaydi" |
| `speech-dataset-generator` | ❌ faqat Ubuntu 22 |

Qulay zaxira yo'l: **`Purfview/whisper-standalone-win`** — `.exe` bilan keladi, Python kerak emas.

### Rad etilganlar va sababi

| Vosita | Nega yaramaydi |
|---|---|
| `whisper-timestamped` | Ma'lum matnni qabul qilmaydi (!), AGPL-3.0 |
| WhisperX (yolg'iz) | O'zbek align modeli yo'q → `ValueError` |
| Gentle | Faqat ingliz tili |
| `audiogrep` | Aligner emas, arxivlangan |
| `stable-ts` | `--align` bilan ishlaydi, MIT — lekin **2026-05-30 da arxivlangan** |

### Matn kuchli mos kelmasa

DSAlign algoritmini **ko'chirib olish** (DSAlign'ning o'zini emas — DeepSpeech o'lgan): rekursiv divide-and-conquer anchor + Smith-Waterman, `match 100 / mismatch −100 / gap −100`, 3-gram prefilter. Bu Moreno et al. (1998) anchor rekursiyasining zamonaviy avlodi.

### Qayerdan kesish kerak — uch signalni birlashtirish

```
1. Gap chegarasi      → NFA default: ['.', '?', '!', '...']   (NeMo yana ":|;" qo'shadi)
2. So'z timestamp'lari → 20 ms aniqlik (MMS'da 50 frame/sek)
3. Silero VAD sukunati → yakuniy snap
```

MMS'ning `get_spans` funksiyasi segmentlar orasidagi pauzani **o'rtasidan** bo'ladi:
`int((prev_seg.start + prev_seg.end) / 2)` — shu tufayli kliplar tabiiy boshlanishini saqlaydi.

**Maqsad uzunlik:** 2–12 s (ManaTTS retsepti). Taqqoslash uchun LJSpeech realligi: **1.11–10.10 s, o'rtacha 6.57 s**.

---

## 4. Sifat filtri — aniq chegaralar bilan

### Bosqichma-bosqich zanjir

**0. Master tayyorlash**
```
Native rate (48/44.1 kHz), mono, 16-bit PCM
DC offset olib tashlash → polarity tekshiruvi  (LibriTTS: "DC offsets are positive")
Clipping: ≥3 ketma-ket full-scale sample = flattened run
         >0.1% sample rail'da  →  RAD
Global loudness: −23 LUFS, TP ≤ −1 dBTP
ffmpeg loudnorm two-pass  linear=true    ← dynamic = kompressiya = TTS uchun zarar
```

**1. Denoise — QILING (dalil kuchli)**

LibriTTS-R eksperimenti hal qiluvchi: restoration qilingan audioda o'qitilgan TTS **MOS 2.59–3.35 → 4.06–4.31**.

Vosita: Demucs/HTDemucs yoki DeepFilterNet3 (48 kHz).

⚠️ Lekin 640 sampldan 2 tasi (**~0.3%**) restoration xatosi bilan buzilgan va **aniqlanmagan**. Shuning uchun majburiy tekshiruv:
```
MCD(original, enhanced) ≤ 5 dB     — oshsa ovoz o'zgargan, TASHLA
```

**2. VAD kesish** — silero-vad (16 kHz analiz nusxasida)
```python
threshold=0.5
min_speech_duration_ms=250
min_silence_duration_ms=250-300      # default 100 juda agressiv
speech_pad_ms=250                    # NeMo Curator 300; TTS uchun 400-500 ham asoslangan
max_speech_duration_s=20
```
- Segment **3–20 s**
- Gap **< 0.55 s** bo'lsa merge qil (WenetSpeech4TTS)
- Ichki uzluksiz sukunat **> 930 ms** → tashla (GLOBE)

**3. Obyektiv sifat**
```
DNSMOS P.835 OVRL ≥ 3.0   (Emilia)
UTMOS            ≥ 3.5
```
⚠️ **Manbalar ziddiyatda:** WenetSpeech4TTS ≥4.0/3.8/3.6 talab qiladi; in-the-wild baholash esa **2.7** ni eng yaxshi topgan ("denoising + permissive filtering = best compromise"). NeMo Curator TTS uchun UTMOS 4.0 tavsiya qiladi — lekin bu faqat ko'p ma'lumot bo'lganda.
👉 **Kichik datasetda 2.7–3.0 dan boshlang.**

**4. SNR**
```
WADA-SNR ≥ 20 dB        (LibriTTS "clean" mezoni)
```
IndicVoices-R 25 dB, GLOBE 0 dB ishlatadi — 20 dB oraliq, oqilona qiymat.

**5. ASR back-check — ENG MUHIM FILTR**
```
Qayta transkripsiya → normalizatsiya → solishtirish

CER ≤ 5%      →  QABUL           (IndicVoices-R)
CER 5–15%     →  QO'LDA KO'RISH
CER > 15%     →  RAD

Chegaralarda yumshoqroq:  edge CER ≤ 60%  (ilk/oxirgi 5 belgi)
```
⚠️ ASR'ning o'z CER floor'i bor (~2%, ParsVoice) — undan pastga tushirmang.
⚠️ Yumshoq mezonlar (CER 20–30%) **ASR uchun**, TTS uchun emas.
⚠️ **O'zbekda bu qadamni Whisper bilan qilmang** — 5-bo'limga qarang.

**6. Alignment confidence**
```
ctc-segmentation score > −2       (NeMo run_segmentation.sh MIN_SCORE=-2)
```
MFA skorlariga ishonmang — rasmiy hujjat aniq chegara bermaydi.

**7. Speaker tekshiruvi**
```
ECAPA-TDNN embedding → korpus medoidiga cosine ≥ 0.65
eng past 10% ni qo'lda ko'rishga belgila   (WenetSpeech4TTS uslubi)
```

**8. Chiqarish**
```
Masterdan BITTA qadamda resample, soxr_vhq
  22050 Hz → VITS
  24000 Hz → XTTS / StyleTTS2
Sukunatni nolga kes → har faylga BIR XIL 100 ms oxirgi pad  (StyleTTS2)
```
👉 Pad **miqdori** emas, **mutanosibligi** muhim. Nomutanosib trim → model nafas/shovqin chiqarishni o'rganadi.

### NeMo'ning tayyor filtr konfiguratsiyasi (ko'chirib olish uchun)

`run_filter.sh`:
```
CER_THRESHOLD=30
WER_THRESHOLD=75
CER_EDGE_THRESHOLD=60
EDGE_LEN=5
LEN_DIFF_RATIO_THRESHOLD=0.3
MIN_DURATION=1
MAX_DURATION=20
```

### Windows muammolari
- `ctc-segmentation` — wheel yo'q, MSVC Build Tools kerak
- `resemble-enhance` — deepspeed talab qiladi → WSL2
- Qolganlari pip bilan ishlaydi

---

## 5. O'zbek tili: nima BOR, nima YO'Q

### Mavjud datasetlar (haqiqiy raqamlar)

| Dataset | Hajm | Spiker | Yozuv | Litsenziya |
|---|---|---|---|---|
| **FeruzaSpeech** ⭐ | **59.61 soat** (kitob 21.57 + BBC Uzbek 38.04) | **1** (toshkentlik ayol) | Lotin **va** Kirill | CC BY 4.0 |
| USC (Uzbek Speech Corpus) | **105 soat** / 958 spiker | ko'p | — | ISSAI CC BY 4.0 vs HF MIT (ziddiyat) |
| UzbekVoice (Mohir.ai) | ~900 soat | ko'p | Lotin | ko'rsatilmagan |
| uzbekvoice-filtered | 503k qator | ko'p | — | **Apache 2.0** |
| Common Voice uz (cv-25.0) | 265.75 soat yozilgan / **101.01 validated** | 2,299 | Lotin | CC-0 |
| FLEURS uz_uz | ~12 soat | ko'p | Lotin | CC BY 4.0 |
| VoxPopuli | ❌ o'zbek **yo'q** | — | — | — |

⚠️ **Keng tarqalgan xato:** internetda USC "900 soat" deb yuradi. Bu **noto'g'ri** — 105 soat / 958 **spiker** (958 raqami spikerlar soni). "~900 soat" aslida UzbekVoice.

⭐ **FeruzaSpeech nega muhim:** bitta spiker, kitob o'qigan, ikki yozuvda transkripsiya, tijoratga ruxsat beruvchi litsenziya. Uning **kitob qismi 21.57 soat** — bu 0 dan VITS o'qitish uchun kerakli "20–25 soat" oynasiga aynan tushadi.

### 🔴 Whisper o'zbek uchun YARAMAYDI

```
Whisper zero-shot o'zbek WER = 90.2%
```

Bu ASR back-check filtrini butunlay buzadi. Ishlatish kerak:

| ASR | Ko'rsatkich | Litsenziya |
|---|---|---|
| **Kotib** (sizda bor) | CER **7.0%** | — |
| GigaAM Large | Feruza 5.93% WER | — |
| ROVER (ansambl) | 5.15% | — |
| `nvidia/stt_uz_fastconformer_hybrid_large_pc` | FLEURS DEV 17.52% (punktuatsiyasiz 12.20%) | **CC-BY-4.0**, Lotin **va** Kirill, punktuatsiya bilan |

### Tayyor vositalar (mavjud, ishlatish kerak)

| Vazifa | Vosita | Litsenziya |
|---|---|---|
| Raqam→so'z (butun/kasr/valyuta/foiz/rim) | **`uztextnorm` 0.5.3** | MIT |
| Apostrof kanonlashtirish (**12 o'xshash belgi** → U+02BB/U+02BC) | **`uztext`** | MIT |
| G2P | espeak-ng `uz` (status: *testing*), epitran `uzb-Latn` + `uzb-Cyrl` | — |
| Lotin→IPA jadvali (eng to'liq) | IS2AI TurkicTTS `ipa_convert.py` + `ipa_mapping.pdf` | — |

⚠️ **CharsiuG2P `uzb` bor, lekin mualliflarning o'zi "do not use it" deydi** — noto'g'ri.

### ❌ O'zbek uchun MAVJUD EMAS

- MMS-TTS **Lotin** o'zbek — faqat `mms-tts-uzb-script_cyrillic`, va u **CC-BY-NC** (tijorat taqiq)
- Piper o'zbek ovozi
- XTTS-v2 o'zbek
- NeMo-text-processing o'zbek
- O'zbek talaffuz lug'ati (pronunciation dictionary)
- Ko'p spikerli ochiq TTS korpusi

### ⚠️ Sizning mavjud pipeline'ingizdagi muammo

`AI_studio/assistant/transliterate.py` Lotin→Kirill o'girib MMS-TTS'ga beradi. Tadqiqot:

> Lotin→Kirill transliteratsiya **F1 = 0.89** — ~**10 so'zdan 1 tasi xato**. 233 ta istisno so'z, asosan ruscha o'zlashmalar.

Bu eng zaif bo'g'in. Ikki alternativa:
1. **`facebook/mms-tts-tur`** (turkcha) dan transfer — Turkman tilida bu usul ishlagani tasdiqlangan
2. MMS-TTS tokenizer vocab'ini Lotin belgilariga qayta moslash (embedding layer almashtirish, fine-tune vaqtida)

### O'zbek tilining xususiyatlari

| Xususiyat | Ta'siri |
|---|---|
| **Unli harmoniya deyarli yo'q** (Ido 2025, JIPA) | 🟢 Turkcha/qozoqchadan sodda — ishni osonlashtiradi |
| Digraflar `sh`, `ch`, `ng` | 🟡 2 belgi = 1 fonema — belgi darajasidagi CTC va G2P uchun muammo |
| Apostrof xaosi: `oʻ` `o'` `o‘` `ō`, `gʻ` `g'` | 🔴 12 xil variant real matnda — `uztext` bilan kanonlashtirish shart |
| 2026 alifbo islohoti (Senat 2026-09-10: `Ö Ğ Ş Ç`, 28 harf) | 🟢 Digraf muammosini hal qiladi |

### Miqdor dalillari

- Darija: **1.2 soat** + 12 soat ko'p tilli pretraining → DNSMOS 3.72
- Yangi trifonlar **~5,000 gapda to'yinadi**
- 20 soat ≈ 10,000 gap
- MMS-TTS + TurkicTTS zero-shot o'zbek MOS **2.85** (qozoq in-language 4.18 bilan solishtiring)

### O'zi qurish kerak bo'lgan narsalar

- Sana / vaqt / telefon raqami normalizatsiyasi
- Qisqartma kengaytirish
- Ruscha/inglizcha loan-word talaffuzi
- Greedy trifon gap-tanlash skripti
- Homoglif tozalash
- espeak/epitran aniqligini qo'lda validatsiya

---

## 6. Mavjud vositalar: nimani qayta ishlatish, nimani qurish

### 🎯 Asosiy strukturaviy topilma

**Hech bir mavjud vosita "to'liq matn + uzun audio → LJSpeech" ni Windows'da yaxshi qilmaydi.** Ekosistema ikkiga bo'lingan va ikki yarim uchrashmaydi:

```
1-YARIM: Whisper asosidagi generatorlar
  ozen-toolkit, speech-dataset-generator,
  tts-dataset-generator, speech-splitter
  → Sizning eng katta boyligingizni — haqiqiy matnni — TASHLAB YUBORADI
  → Uning o'rniga Whisper xatolarini datasetga yozadi

              ⬇  BO'SHLIQ AYNAN SHU YERDA  ⬇

2-YARIM: Audiokitob alignerlari
  easyaligner, afaligner, syncabook, Storyteller, timething, aeneas
  → Matnni to'g'ri ishlatadi
  → Lekin chiqishi EPUB3/SMIL/JSON sync-map — TTS segmentlari EMAS
```

Siz aynan shu bo'shliqni to'ldirasiz. Bu loyihaning qiymati shunda.

### ♻️ QAYTA ISHLATISH

| Nima | Nima uchun | Holat |
|---|---|---|
| **`kb-labb/easyaligner`** ⭐ | Alignment yadrosi: **soatlarni bir o'tishda**, chunking'siz; so'z bo'yicha confidence; MMS-1B ko'p tilli. RixVox-v2 (23k soat) shu bilan qurilgan | MIT, **tirik** (2026-07) |
| NVIDIA NFA | Zaxira aligner | tirik |
| `stable-ts` ning `refine()` / `regroup()` | Segment chegaralarini yaxshilash mantiqi | MIT, **arxivlangan** — fork qilish mumkin |
| `audio-slicer` RMS algoritmi | Sukunat aniqlash | arxivlangan, algoritm foydali |
| YAMNet | Musiqa/shovqin bo'laklarini filtrlash | tirik |
| DataSpeech metrika ro'yxati | Qaysi ko'rsatkichlarni o'lchash | 2024-09 dan beri jim |
| **NeMo SDP** manifest-per-stage arxitekturasi | Pipeline strukturasi namunasi | tirik |
| **Lhotse** `SupervisionSegment` sxemasi | Ma'lumot modeli | tirik |
| **`idiap/coqui-ai-TTS`** | `find_unique_chars.py`, `compute_statistics.py`, AnalyzeDataset, CheckSpectrograms | **tirik fork** (asl `coqui-ai/TTS` 2024-08 da o'lgan) |

### 🔨 O'ZI QURISH

1. **Alignment → LJSpeech segmenteri**, TTS cheklovlari bilan: 1–15 s, **hech qachon so'z o'rtasidan kesmaslik**, pauzaga buffer qo'shish — lekin qo'shni so'zga **hech qachon** kirmaslik
2. **Confidence bo'yicha saralangan review UI** — klaviatura birinchi o'rinda
3. **Content-hash resume** — 10 soatlik ish yarmida to'xtasa qaytadan boshlamaslik
4. **Windows-native o'rnatish**
5. **Xotira-xavfsiz I/O:** `soundfile.blocks()` + ffmpeg `-ss/-t`
   ⚠️ `librosa.load` 10 soatlik 48 kHz stereo faylda **~14 GB** RAM yeydi — sizda 15.9 GB bor, ya'ni bu bitta xato butun tizimni o'ldiradi

### ☠️ Tashlab ketilgan loyihalar (ishlatmang)

| Loyiha | Oxirgi faoliyat | Qo'shimcha |
|---|---|---|
| aeneas | funksional commitlar **2017** da to'xtagan | AGPL-3.0 |
| ozen-toolkit | 2023-08 | **litsenziyasi umuman yo'q** |
| Mangio-RVC-Fork | 2023-09 | — |
| coqui-ai/TTS | 2024-08 | `idiap/coqui-ai-TTS` forkini ishlatish |
| DeepForcedAligner | arxivlangan 2025-08 | — |
| dataspeech | 2024-09 | — |
| timething | 2024-12 | — |
| audio-slicer | 2024-06 | — |
| stable-ts | arxivlangan 2026-05 | MIT, fork qilish mumkin |

### 👥 Qo'lda ko'rish bosqichi

Bu bosqichdan qutulib bo'lmaydi, lekin qisqartirish mumkin. Vositalar: Label Studio, Doccano, Audino, ELAN, Praat, Audacity labels.

**Kalit g'oya:** avtomatik metrikalar chegara to'g'riligini **to'liq kafolatlamaydi**. Shuning uchun tizimning markaziy vazifasi — "alignment'ni mukammal qilish" emas, balki **"qaysi segmentga ishonmaslik kerakligini aniq ko'rsatish"**:
- alignment confidence
- ASR-vs-reference divergensiyasi (NeMo SDP `ASRInference` uslubida)
- chegara energiyasi
- davomiylik outlier'lari

👉 Natija: 10 soat audio uchun review **10+ soatdan 1–3 soatga** tushadi.

---

## 7. "0 dan o'qitish" realligi

### Modellar nimada o'qitilgan (haqiqiy raqamlar)

| Model | Ma'lumot | GPU | Qadamlar |
|---|---|---|---|
| Tacotron 2 | 24.6 soat | — | — |
| FastSpeech 2 | 17.02 soat | 1× V100 | — |
| **VITS** | ~24 soat (LJSpeech) | **4× V100** | 800k |
| StyleTTS 2, Matcha-TTS | ~24 soat (LJSpeech) | — | — |
| XTTS | 27,281.6 soat | 4× A100 | ~2.5M |
| F5-TTS | **95,000 soat** | 8× A100 | 1 haftadan ko'p |

### Kritik chegaralar

```
< 2 soat    →  alignment umuman shakllanmaydi (WER > 133%) — model gapirmaydi
~10 soat    →  minimal, xavfli
20–25 soat  →  ishonchli (0 dan klassik model uchun)
30–50 soat  →  ideal
```

Yozib olish rejasi: **30–60 soat xom audio** (15–40% rad etiladi).

### Fine-tune vs 0 dan — koreys tilida aniq taqqoslash (12.65 soat)

| | CER | WER | NMOS |
|---|---|---|---|
| 0 dan | 17.3 | 45.9 | 2.64 |
| **fine-tune** | **5.67** | **12.03** | **3.58** |

⚠️ **Ziddiyat:** bitta manba *yangi fonemalarni* o'zlashtirish uchun 0 dan o'qitish har qanday ma'lumot hajmida fine-tune'dan yaxshi ekanini topgan. O'zbek uchun bu muhim bo'lishi mumkin.

### Xulosa sizning holatingiz uchun

| Maqsad | Realistikmi |
|---|---|
| VITS / FastSpeech2 / Matcha-TTS, bitta ovoz, 0 dan | ✅ **Ha**, 20–25 soat toza audio bilan |
| F5-TTS / XTTS klassi, 0 dan | ❌ **Yo'q** — 27k–95k soat, 4–8× A100 |
| 20 soatdan kam bo'lsa | 👉 **Fine-tune qiling** — nashr etilgan farq hal qiluvchi |

---

## 8. Sizning imkoniyatlaringiz (muhit auditi, 2026-09-29)

| Resurs | Holat | Ma'nosi |
|---|---|---|
| GPU | RTX 3050, **8 GB VRAM** | 🟡 Dataset uchun yetarli, o'qitish uchun kam |
| CPU / RAM | 16 yadro / **15.9 GB** | 🟡 `librosa.load` tuzog'iga ehtiyot bo'ling (yuqorida) |
| Disk C: | 46 GB bo'sh (**94% to'la**) | 🔴 Dataset uchun yaramaydi |
| Disk D: | 85 GB bo'sh, `HF_HOME=D:\hf_cache` | 🟢 Ish shu yerda bo'lsin |
| PyTorch | `2.6.0+cu124`, CUDA ✅ | 🟢 Baza tayyor |
| Paketlar | `openai-whisper`, `librosa`, `soundfile`, `pydub`, `transformers 4.57.3`, `datasets` | 🟢 Yarim yo'l bosilgan |
| **Kotib STT** | Ishlaydi (CER 7.0%) | 🟢🟢 **Eng katta ustunlik** — Whisper o'zbekda 90.2% WER |
| MMS-TTS uzb | Ishlaydi | 🟡 Faqat Kirill, CC-BY-NC |
| RVC | `mening_ovozim_e100_s19500.pth` | 🟢 Matn kerak emas — alohida yo'l |

### 🔴 Tuzatish kerak

```
torch 2.6.0  +  torchaudio 2.9.1   → OSError: [WinError 127]
import torchaudio  UMUMAN ISHLAMAYDI

pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Shuningdek `ffmpeg` PATH'da yo'q (asosiy Python 3.10 muhitida `imageio_ffmpeg` ham o'rnatilmagan).

### 8 GB VRAM haqida halol gap

VITS'ning o'zi **4× V100 (har biri 32 GB)** da 800k qadam o'qitilgan. RTX 3050'da kichik batch bilan bu **haftalar–oylar**. Dataset tayyorlash — to'liq mahalliy; o'qitish bosqichida bulutli GPU (RunPod / Vast.ai) rejalashtirilishi kerak.

---

## 9. Tavsiya etilgan arxitektura

```
KIRISH:  to'liq matn (.txt/.epub)  +  uzun audio (.wav/.mp3, soatlar)
   │
   ├─[0]─ MATN TAYYORLASH
   │      uztext (apostrof kanonlashtirish, 12 variant)
   │      uztextnorm (raqam→so'z)
   │      o'zi: sana/vaqt/telefon/qisqartma
   │      → normalizatsiyalangan matn + asl matn (ikkisi saqlanadi)
   │
   ├─[1]─ AUDIO MASTER
   │      native rate, mono, 16-bit · DC offset · clipping tekshiruvi
   │      −23 LUFS, TP ≤ −1 dBTP, loudnorm linear=true
   │
   ├─[2]─ DENOISE (ixtiyoriy lekin tavsiya)
   │      DeepFilterNet3 / HTDemucs
   │      MCD(orig, enhanced) ≤ 5 dB  ← majburiy tekshiruv
   │
   ├─[3]─ FORCED ALIGNMENT            ★ YADRO
   │      ctc-forced-aligner (MMS-300m)  yoki  kb-labb/easyaligner
   │      --window_size 30 --context_size 2 --star_frequency edges
   │      → so'z darajasidagi timestamp + confidence (20 ms aniqlik)
   │
   ├─[4]─ SEGMENTLASH                 ★ O'ZI QURISH
   │      gap chegarasi (.?!) + so'z timestamp + silero VAD snap
   │      pauza o'rtasidan bo'lish: int((prev.start + prev.end)/2)
   │      2–12 s · so'z o'rtasidan kesmaslik · qo'shni so'zga kirmaslik
   │
   ├─[5]─ FILTR ZANJIRI
   │      alignment score > −2
   │      WADA-SNR ≥ 20 dB
   │      DNSMOS OVRL ≥ 3.0  ·  UTMOS ≥ 3.5   (kichik datasetda 2.7 dan boshla)
   │      ASR back-check KOTIB bilan:  CER ≤5% qabul │ 5–15% ko'rish │ >15% rad
   │      ECAPA-TDNN cosine ≥ 0.65
   │      ichki sukunat > 930 ms → tashla
   │
   ├─[6]─ REVIEW UI                   ★ O'ZI QURISH
   │      confidence bo'yicha saralangan · klaviatura bilan
   │      "qaysiga ishonmaslik kerak" ni ko'rsatadi
   │
   └─[7]─ CHIQISH
          soxr_vhq bilan bitta qadamda 22050 Hz (VITS)
          bir xil 100 ms oxirgi pad
          wavs/ + metadata.csv (LJSpeech)  +  rad etilganlar hisoboti
```

**Kesib o'tuvchi talablar:** content-hash resume · `soundfile.blocks()` bilan xotira-xavfsiz I/O · har bosqichda manifest (NeMo SDP uslubi) · barchasi D: diskda.

---

## 10. Amaliy reja

| Bosqich | Ish | Natija |
|---|---|---|
| **0** | `torchaudio==2.6.0` tuzatish, ffmpeg, ish papkasini D: ga ko'chirish | Muhit ishlaydi |
| **1** | **FeruzaSpeech'ni yuklab olish** (59.61 soat, CC BY 4.0) | Tayyor ground truth — pipeline'ni shunda sinash mumkin |
| **2** | `ctc-forced-aligner` + Kotib bilan **1 soatlik prototip** | Yondashuv ishlashini tasdiqlash |
| **3** | Segmenter + filtr zanjirini qurish | To'liq pipeline |
| **4** | FeruzaSpeech'da validatsiya — sizning natijangiz ularning transkripsiyasiga mos keladimi? | **Aniq sifat o'lchovi** |
| **5** | Review UI | Qo'l mehnati 1–3 soat/10 soat |
| **6** | O'z ovozingizni yozish (30–60 soat xom) | 20–25 soat toza dataset |
| **7** | Fine-tune (mahalliy) yoki 0 dan (bulutli GPU) | O'zbek TTS |

**2-bosqichdan boshlang.** Katta rejadan oldin 1 soatlik audioda uchidan-uchiga ishlatib ko'rish — eng katta xatarni eng arzon narxda yopadi.

---

## 11. Eng katta tuzoq (boshqalar boshidan o'tkazgan)

`rhasspy/piper` muhokamasi #191, foydalanuvchi ZachB100:

> *"Whisper completely missed most of the text in the first file"*
> *"nothing about this process is quick and easy other than the training"*

👉 **Bo'g'iz GPU emas, ma'lumot tayyorlash.**

Whisper haqida muhim texnik detal: **sukunatda gallyutsinatsiya qiladi va buni yuqori ishonch bilan qiladi** — past `no_speech_prob` + yuqori `avg_logprob`, ya'ni uning o'z filtri bunday holatni ushlamaydi. Va so'z timestamp'lari — inference vaqtidagi hiyla, pauzalarda buziladi.

Sizning qo'lingizda haqiqiy matn borligi bu muammoning **yarmini** yo'q qiladi. Qolgan yarmi saqlanadi: **hech bir avtomatik metrika chegara to'g'riligini to'liq kafolatlamaydi.**

Shuning uchun tizimning shiori:

> ### "Alignment'ni mukammal qilish" emas — **"qaysi segmentga ishonmaslik kerakligini aniq ko'rsatish."**

---

## 12. Batafsil manbalar

`tadqiqot/` papkasida (320 KB, 3447 qator):

| Fayl | Mazmuni |
|---|---|
| `research_1_alignment.md` | Forced alignment vositalari, to'liq taqqoslash, Windows holati, algoritmlar |
| `research_2_corpora.md` | Korpus qurish metodologiyasi, statistika, "0 dan" talablari, muvaffaqiyatsizlik keyslari |
| `research_3_quality.md` | 63 qatorli sifat metrikalari jadvali, barcha chegaralar manbalari bilan |
| `research_4_uzbek.md` | O'zbek resurslari, G2P, normalizatsiya, alifbo, miqdor dalillari |
| `research_5_tools.md` | ~28 tekshirilgan vosita, tashlab ketilganlar, arxitektura namunalari |
| `urls_all.txt` | Barcha manba havolalari |

**Halol eslatma:** hisobotlarda ~12+ joyda "ma'lumot topilmadi" deb ochiq yozilgan. Eng muhimlari: hech bir aligner uchun nashr etilgan RTF (realtime factor) raqamlari yo'q; MFA va MMS_FA uchun millisekund aniqligi o'lchovlari yo'q; `kalpy` win-64 mavjudligi tasdiqlanmagan. Bu joylarni taxmin bilan to'ldirmadik.

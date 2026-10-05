      # AI Dataset Maker — natijalar va holat

**Oxirgi yangilanish:** 2026-10-02
**Holat:** ✅ tizim ishlaydi · ✅ haqiqiy dataset yasadi · ✅ chegaralar kalibrlandi
**Kod:** 1,413 qator (7 fayl)

---

## 1. Tizim nima qiladi

```
KIRISH:   uzun o'zbek audio (soatlar)  +  o'sha audioning to'liq matni
CHIQISH:  LJSpeech formatidagi TTS dataset (wavs/ + metadata.csv)
```

Matn bo'lmasa — Kotib ASR bilan yasash mumkin (`--asr-text`), lekin sifat
sezilarli pasayadi (5.2-bo'limga qarang).

---

## 2. 🎯 Asosiy natija — o'lchangan

**Sinov:** FLEURS `uz_uz` dan 10.11 daqiqa (64 klip ulangan, **qo'lda yozilgan
ground-truth matn** bilan, 965 so'z).

| Ko'rsatkich | Natija |
|---|---|
| So'z moslashtirildi | **977 / 977** |
| Alignment tezligi | **RTF 0.027** — 606 sek audio 16 sekundda |
| Segment yasaldi | 68 |
| **Yakuniy dataset** | **51 klip · 0.109 soat · 7.67 sek o'rtacha** |
| **Saqlanish foizi** | **75%** |
| **Qabul qilinganlar CER** | **median 2.11%** · o'rtacha 2.43% · max 9.24% |
| CER = 0 bo'lganlar | 21/51 (**41%**) |
| Rad etilganlar CER | median 14.7% |

### Saqlanish foizini nashr etilganlar bilan solishtirish

| Korpus | Saqlanish |
|---|---|
| Libriheavy | ~83% |
| HiFiTTS-2 | 82% |
| **bizning tizim** | **75%** |
| EuroSpeech | 65.4% |
| WenetSpeech4TTS | ~56% |
| Emilia | 29.43% |
| GLOBE | 16% |

👉 Qabul/rad CER'i aniq ajralgan (**2.11% vs 14.7%**) — filtr ishlayapti.

### Kalibratsiya bosqichlari (ortiqcha moslashtirmaslik uchun 3 ta bilan to'xtatildi)

| Versiya | O'zgarish | Klip | Soat | Saqlanish | CER median |
|---|---|---|---|---|---|
| v1 | CER 5% · `min_score` 0.25 | 34 | 0.072 | 52% | 0.00% |
| v2 | CER 10% · `mean_score` 0.35 | 48 | 0.102 | 70.6% | 2.02% |
| **v3** | toza CER ball darvozasini chetlab o'tadi | **51** | **0.109** | **75%** | **2.11%** |

---

## 3. Arxitektura

```
ai dataset maker/
├── main.py              184 qator   CLI, 6 bosqichli quvur
└── src/
    ├── uz_text.py       273 qator   o'zbek matn qatlami
    ├── align.py         172 qator   MMS_FA forced alignment
    ├── segment.py       230 qator   2-12 sek kesish + 4 filtr
    ├── quality.py       278 qator   Kotib ASR back-check, CER, SNR
    ├── vad.py            94 qator   silero-VAD chegara snap
    └── export.py        182 qator   LJSpeech eksport
```

### Quvur

```
[1] audio yuklash          16 kHz mono
[2] matn                   fayldan yoki Kotib ASR dan
      ↓ uz_text.split_aligned()  → dataset matni (ʻ ʼ) + align matni (a-z + ')
[3] alignment              MMS_FA, bo'laklab (20 sek + 2 sek ustma-ust)
      ↓ so'z darajasida timestamp + score · keyin MMS_FA VRAM'dan bo'shatiladi
[4] segmentlash            gap chegarasida + silero-VAD snap, 2-12 sek
      ↓ 4 filtr: uzunlik, ichki sukunat, so'z cho'zilishi
[5] ASR back-check         Kotib, batch (OOM'da o'zi yarmiga tushadi) → CER
[6] eksport                22050 Hz (VITS) yoki 24000 Hz (XTTS)
```

---

## 4. Ishlatish

```bash
# matn bor (TAVSIYA ETILADI)
python main.py --audio kitob.wav --text kitob.txt --out dataset/

# matn yo'q (sifat sezilarli pasayadi)
python main.py --audio ovoz.wav --asr-text --out dataset/
```

| Parametr | Default | Izoh |
|---|---|---|
| `--sr` | 22050 | 22050=VITS/LJSpeech · 24000=XTTS/StyleTTS2 |
| `--star` | edges | `none`/`edges`/`between` — matn qanchalik mos kelmaydi |
| `--chunk` | 20 | alignment bo'lak uzunligi (VRAM uchun) |
| `--batch` | 4 | ASR batch. 8 GB uchun 2-4; OOM'da o'zi tushadi |
| `--no-vad` | — | VAD snap'ni o'chirish |
| `--no-asr-check` | — | **10× tez**, lekin eng kuchli filtr o'chadi |
| `--device` | auto | `cuda` yoki `cpu` |

### Chiqish fayllari

| Fayl | Mazmuni |
|---|---|
| `wavs/0001.wav ...` | kliplar, 16-bit PCM |
| `metadata.csv` | `0001\|matn` — LJSpeech |
| `segments.csv` | `nom\|start\|end\|min_score\|cer\|snr_db` — provenance |
| `rejected.csv` | `vaqt\|sabab\|cer\|mean_score\|matn` |
| `report.json` | statistika, saqlanish foizi |

### Tezlik (10 daqiqa audio, RTX 3050)

```
alignment        16 sek
VAD + segment     3 sek
ASR back-check  231 sek    ← eng sekin qadam (65 segment, batch=2)
eksport           2 sek
                ───────
jami            ~268 sek   (--no-asr-check bilan: 19 sek)
```

---

## 5. ✅ NIMA ISHLAYDI

### 5.1 O'zbek matn qatlami

Sinov: `python src/uz_text.py`

| Kirish | Chiqish | Kim qiladi |
|---|---|---|
| `1 250 000,50 so'm` | `bir million ikki yuz ellik ming soʻm ellik tiyin` | **bizning kod** |
| `2026-yil 10-sentyabrda` | `ikki ming yigirma oltinchi yil oʻninchi sentyabrda` | uztextnorm |
| `15%` · `3/4` · `XXI` | `oʻn besh foiz` · `toʻrtdan uch` · `yigirma birinchi` | uztextnorm |
| `+998 90 123 45 67` | `plus toʻqqiz toʻqqiz sakkiz toʻqqiz nol bir ikki uch...` | **bizning kod** |
| `Soat 14:30` | `soat oʻn toʻrt oʻttiz` | **bizning kod** |
| `prof. A. Qodirov va h.k.` | `professor a. qodirov va hokazo` | **bizning kod** |
| `25 km, 3 kg` | `yigirma besh kilometr, uch kilogramm` | **bizning kod** |
| `$150 edi` | `bir yuz ellik dollar edi` | **bizning kod** |
| `O\`zbekiston san'at g'alaba` | `oʻzbekiston sanʼat gʻalaba` | uztext |

### 5.2 Alignment

- **977/977 so'z**, RTF 0.027
- **Chunking siljish bermaydi** — tasdiqlangan: chunk 10/20/40 sek → bir xil
  natija (1-so'z 0.579/0.579/0.580 sek, oxirgi 604.98 sek, fps 50.0-50.1)
- O'zbek uchun maxsus akustik model kerak emas (MMS, 1,130 til)

### 5.3 Filtrlar yolg'on ma'lumotni ushlaydi

Kotib takrorlanish gallyutsinatsiyasi qilgan joyda:

```
Haqiqiy so'zlar          score 0.313 · 0.326 · 0.313
Gallyutsinatsiya         score 0.008 · 0.003 · 0.001     ← ~40× farq
```

Va bitta `toʻxtang` **3.16 sekundga** cho'zilgan → `MAX_WORD_DUR` ushladi.

FLEURS sinovida CER ajralishi: **qabul 2.11% · rad 14.7%**.

### 5.4 Ikkita manba kutubxona xatosi aniqlandi va chetlab o'tildi

```
"1 250 000,50 so'm"
  uztextnorm →  "bir yuz yigirma besh million ellik soʻm"  ❌ (125 million!)
  uztext     →  "bir million ikki yuz ellik ming soʻm ellik tiyin"  ✅
```

Tadqiqotga tuzatish: `uztextnorm.dates.expand_dates` **mavjud** (tadqiqot "yo'q,
o'zi qurish kerak" degan edi — xato).

---

## 6. ❌ NIMA ISHLAMAYDI / CHEKLOVLAR

### 6.1 `ovoz_1.wav` / `ovoz_2.wav` TTS dataseti uchun yaroqsiz

Sinov natijasi: **0 klip**. Sabab — manba materialda:

```
"a net, toʻxtang, talant sizga qarab yuribdi"
"otil aka, meni urdingiz, toʻqson besh qoldi"
"nima, nima, nima, chota? kiring, kiring"
```

Kino/komediya dialogi — ko'p odam, baqiriq, ruscha aralash.
RVC uchun ✅ yetarli · TTS dataseti uchun ❌ yaroqsiz.

### 6.2 `--asr-text` ishonchsiz

Kotib Whisper arxitekturasida — **takrorlanish gallyutsinatsiyasi** qiladi.
13 segmentdan hammasi rad etilgan edi. **Har doim haqiqiy matn bilan ishlatilsin.**

### 6.3 `ctc-forced-aligner` Windows'da o'rnatilmaydi

```
pip install ctc-forced-aligner → error: link.exe ... exit status 1120
```
**Hal qilindi:** `torchaudio.pipelines.MMS_FA` (bir xil MMS modeli, build kerak emas).

### 6.4 ASR back-check — eng sekin qadam

```
alignment:  16 sek  (606 sek audio)
ASR:       231 sek  ← 14× sekinroq
```
Tezlashtirish mumkin: `faster-whisper` (CTranslate2, 3-4×). Hali qilinmagan.

### 6.5 Hal qilingan nosozliklar (keyingi marta takrorlanmasin)

| Muammo | Sabab | Tuzatish |
|---|---|---|
| ASR **timeout** (17 daq, tugamagan) | 7 sek klipga `chunk_length_s=30` → Whisper'ning eksperimental sekin yo'li | 28 sek'dan qisqa audioga berilmaydi |
| **CUDA OOM** (*"14.53 GiB allocated"* 8 GB da) | `batch=16` | Moslashuvchan batch — OOM'da yarmiga tushadi, 1 gacha |
| VRAM to'lishi | MMS_FA + Kotib birga | Alignment'dan keyin MMS_FA bo'shatiladi |
| `torchaudio` `WinError 127` | 2.9.1 vs torch 2.6.0 | `torchaudio==2.6.0+cu124` |
| `UnicodeEncodeError` U+02BB da | Windows cp1252 konsol | `PYTHONIOENCODING=utf-8` |

---

## 7. Chegaralar va manbalari

### Tadqiqotdan (manbasi bor)

| Chegara | Qiymat | Manba |
|---|---|---|
| Segment uzunligi | 2–12 sek | ManaTTS |
| Qattiq maksimum | 20 sek | NeMo `MAX_DURATION` |
| Birlashtirish gap | < 0.55 sek | WenetSpeech4TTS |
| Ichki sukunat | > 0.93 sek → rad | GLOBE |
| Chegara CER | ≤ 60% (ilk/oxirgi 5 belgi) | NeMo `CER_EDGE_THRESHOLD=60` |
| SNR | ≥ 20 dB | LibriTTS "clean" |
| Loudness | −23 LUFS, TP ≤ −1 dBTP | EBU R128 |
| Oxirgi pad | 100 ms, bir xil | StyleTTS2 |
| VAD | `threshold=0.5`, `min_silence_ms=280` | research_3_quality.md |

### 🟢 FLEURS o'lchovi bilan KALIBRLANGAN (2026-10-02)

| Chegara | Qiymat | Asos |
|---|---|---|
| `CER_ACCEPT` | **0.10** | Kotib'ning **o'z** xatosi ~7%. 5% chegara bilan 9 ta yaxshi klip faqat Kotib xatosi uchun belgilangan edi |
| `CER_REVIEW` | 0.20 | shunga mos |
| `MEAN_ALIGN_SCORE` | **0.35** | Qabul qilinganlarda mean median 0.655. `min_score` darvoza bo'lgani NOTO'G'RI edi — bitta qisqa so'z (`u`, `va`) butun segmentni yiqitardi |
| `MIN_ALIGN_FLOOR` | 0.02 | faqat alignment butunlay yiqilganini ushlash uchun |
| **CER ustunligi** | toza CER (≤10%) ball darvozasini **chetlab o'tadi** | 9 ta klip CER=0 bo'lsa ham ball poli bilan rad etilgan edi. ASR matnni aynan eshitsa — klip to'g'ri |
| `PAD_START` / `PAD_END` | **0.35 / 0.55** | O'lchandi: MMS_FA 1-so'zni nutq boshlanishidan **~290 ms keyin**, oxirgisini tugashidan **~480 ms oldin** qo'yadi (CTC "ishonchli yadro"). Nosimmetrik pad shuni qoplaydi |
| `MAX_WORD_DUR` | 1.5 sek | Sinovda 3.16 sek'lik so'z = gallyutsinatsiya belgisi |

### ⚠️ Halol eslatmalar

- **`estimate_snr()` WADA-SNR EMAS** — soddaroq energiya nisbati. Qiymatlarni
  WADA bilan solishtirmaslik kerak.
- Kalibratsiya **bitta 10 daqiqalik FLEURS namunasida** qilingan. Boshqa
  materialda (ayniqsa bitta spikerli uzun kitob o'qishda) qayta tekshirish kerak.
- Chegara aniqligi uchun o'zim yasagan metrika **ishonchsiz chiqdi**: "nutq bormi"
  tekshiruvi qisqa tovushlarga (nafas, chertish) ham yonadi — nutq median
  amplitudasi 0.0047, chegaram 0.0144, topilgan max 0.075. Shuning uchun yakuniy
  hakam — **ASR back-check CER**, sun'iy chegara metrikasi emas.

---

## 8. Muhit

| Komponent | Versiya | Izoh |
|---|---|---|
| Python | 3.10.0 | |
| torch / torchaudio | 2.6.0+cu124 | CUDA ✅ · **versiyalar mos bo'lishi SHART** |
| transformers | 4.57.3 | |
| uztext | 0.1.0 | GitHub'dan (PyPI'da **yo'q**) |
| uztextnorm | 0.5.3 | PyPI |
| silero-vad | 6.2.3 | ulangan, ishlaydi (74 oraliq topdi) |
| soxr / pyloudnorm | 1.0.0 / 0.2.0 | |
| GPU | RTX 3050, 8 GB | dataset uchun yetarli |

```bash
pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install uztextnorm pyloudnorm soxr imageio-ffmpeg silero-vad
pip install "git+https://github.com/abafaboy/uztext#subdirectory=python"

# Windows
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set HF_HOME=D:\hf_cache
```

---

## 9. Hali qilinmagan

| Ish | Nega kerak |
|---|---|
| **Bitta spikerli uzun o'qishda sinash** | FLEURS ko'p spikerli, 4-11 sek kliplardan ulangan. Haqiqiy kitob o'qishi boshqacha |
| `faster-whisper` ga o'tish | ASR back-check 14× sekinroq, eng katta bo'g'iz |
| Denoise bosqichi | LibriTTS-R dalili: MOS 2.59 → 4.06. `MCD ≤ 5 dB` tekshiruvi bilan |
| Review UI | `rejected.csv` da 5 ta "ko'rish" segment qo'lda tekshirishni kutadi |
| Resume (content-hash) | 10 soatlik ish yarmida to'xtasa qaytadan boshlamaslik |
| Iterativ qayta-alignment | WorldSpeech usuli: zaif ASR tillarda +19.5–201% ma'lumot |
| Ko'p faylli rejim | Hozir bitta audio + bitta matn |
| DNSMOS / UTMOS | Tadqiqotda tavsiya etilgan, hali ulanmagan |

---

## 10. Xulosa

| Savol | Javob |
|---|---|
| Tizim ishlaydimi? | ✅ **Ha** — uchidan-uchiga |
| Dataset yasadimi? | ✅ **Ha** — 51 klip, 0.109 soat, 75% saqlanish |
| Sifat qanday? | ✅ CER median **2.11%**, 41% da CER=0 |
| Tez ishlaydimi? | ✅ Alignment RTF 0.027 · to'liq quvur 10 daq audio → 4.5 daq |
| Chegaralar kalibrlanganmi? | ✅ FLEURS ground-truth bilan, 3 iteratsiya |
| Nima kerak? | **Bitta spikerli, toza, matni bor uzun o'qish** |

> **Tizim tayyor va o'lchangan. Keyingi qadam — haqiqiy kitob o'qishida sinash.**

Sinov materiali va natijalar: `scratchpad/fleurs_test/` (vaqtinchalik papkada —
kerak bo'lsa `build_fleurs_test.py` bilan qayta yasaladi).

---

## 11. OmniVoice sinovi — 2026-10-02 (TEKSHIRUV.md #1 bajarildi)

**O'rnatish:** `pip install omnivoice` → 0.2.1. Model `k2-fsa/OmniVoice` 3.27 GB → `D:\hf_cache`.
⚠️ `transformers` 4.57 → **5.18** ga ko'tarildi. Kotib tekshirildi — **ishlaydi** (faqat deprecation warning).

**Sinov-sinov (FLEURS ayol ovozi referens):** 11.5 s referens → 20.1 s sintez, **CER 0.5%**.

**Haqiqiy sinov (foydalanuvchi ovozi, Telegram .ogg, 18.1 s):**

| | |
|---|---|
| Sintez | **28.4 sek** (model referens tezligiga moslashdi) |
| Vaqt | 46 s (RTF 1.62, model yuklash bilan) |
| **Tushunarlilik (Kotib)** | **CER 0.8% — a'lo** |
| O'xshashlik | foydalanuvchi baholaydi — `omnivoice_test/natija/sintez_30s.wav` |

**Skript:** `omnivoice_test/sinov.py <ovoz_fayli>` — .ogg/.mp3/.wav qabul qiladi,
Kotib bilan ref_text yasaydi, OmniVoice sintez, Kotib baho, `hisobot.json`.

**Xulosa:** OmniVoice RTX 3050 da ishlaydi, o'zbekcha a'lo tushunarli. Ovoz
o'xshashligi — subyektiv, foydalanuvchi qarori kutilmoqda.

---

## 12. Haqiqiy audiokitob sinovi — "O'tkan kunlar" (2026-10-02)

**Material:** audiobook.uz, 1-trek, 22 s dan boshlab 9:38 (bitta diktor, badiiy o'qish) +
ziyouz.com PDF matni (1926 orfografiyasi, 707 so'z). Skript: `tests/prep_otkan_kunlar.py`.
Faqat lokal sinov uchun — o'qitishga ishlatilmaydi.

| Versiya | Sozlama | Klip | Soat | Saqlanish |
|---|---|---|---|---|
| v1 | `--max-pause 0.93` (default) | 63 | 0.110 | 76.8% |
| **v2** | `--max-pause 1.5` + rim raqami himoyasi | **76** | **0.134** | **92.7%** |

Alignment 707/707 so'z. Yuqoridagi ikki qator — faqat segmentlash filtrlaridan keyin (ASR'siz).

**To'liq ish, Kotib ASR back-check bilan (GPU bo'sh, `--batch 4 --max-pause 1.5`):**

| | |
|---|---|
| Yakuniy dataset | **72 klip · 0.126 soat** (82 segmentdan, 87.8%) |
| Qabul qilinganlar CER | median **1.70%** · o'rtacha 2.10% · max 8.16% · 24 tasida CER = 0 |
| Rad: juda qisqa | 6 |
| Rad/ko'rish: ASR sabab | 4 (2 tasi audio boshida/oxirida kesilgan gap — to'g'ri ushlandi) |
| Vaqt | jami **140 s** (9.6 daq audio) · ASR qismi 104 s / 76 segment |

Kotib fp16 + batch 4: **1.4 s/segment** (FLEURS sinovida fp32 + batch 2: 3.6 s/segment).

**Shu sinovda topilgan va tuzatilgan:**
- `uztextnorm` katta harfli so'z ichidagi rim harflarini raqam deb o'qiydi:
  `O‘G‘LI` -> `oʻgʻellik birinchi` (LI = 51). Tuzatish: `uz_text.guard_roman()`.
- GLOBE'ning 0.93 s ichki sukunat chegarasi badiiy o'qish uchun qattiq (13/82 segment
  rad etilgan, pauzalar 0.94-1.42 s). Tuzatish: `--max-pause` parametri.
- Kotib endi **fp16** da yuklanadi (VRAM 1.53 GB) + `max_new_tokens` chegarasi.

**Muhit saboqi:** GPU'ni boshqa jarayon band qilsa (bu safar Ollama `qwen3:8b`, 6.2 GB,
97%), Kotib 30x sekinlashadi (7 s audio -> 98 s). Uzoq ishdan oldin `nvidia-smi` va
`ollama ps` ni tekshirish kerak. CPU'da Kotib ~6 GB RAM so'raydi.

---

## 13. 2 soatlik sinov — "O'tkan kunlar" 1-4 treklar (2026-10-05)

**Material:** audiobook.uz 1-4 treklar ulangan, aniq 120 daqiqa, bitta diktor + kitob PDF matni
(14 000 so'z berildi, audioga 9 354 tasi to'g'ri keldi). Faqat lokal sinov.

| | |
|---|---|
| Moslashtirish | 9 354 so'z, 24 ta 5 daqiqalik blok, uzilishsiz |
| Segment | 1 046 (kesish filtridan 951 o'tdi) |
| **Yakuniy dataset** | **916 klip, 1.566 soat**, o'rtacha 6.16 s |
| Saqlanish | segmentlarning 87.6% · audio vaqtining 78% |
| Qabul qilinganlar CER | median **2.22%**, o'rtacha 2.64%, max 10.0%, 232 tasida CER = 0 |
| Vaqt bo'yicha tekislik | har 30 daqiqada 240 / 218 / 219 / 239 klip (siljish yo'q) |
| Jami vaqt | 3 215 s (~54 daq): moslashtirish ~3 daq (170 s), Kotib ~49 daq (2 927 s) |

Rad etilgan 130: juda qisqa 64, ichki sukunat 24, CER 5-20% (qo'lda ko'rish) 22,
alignment yiqilgan 10, so'z cho'zilgan 7, CER > 20% 3.

**Shu sinov uchun qo'shilgan:**
- `align_long()` - blokli moslashtirish. Bir yo'la moslashtirish xotirasi (audio x matn) ga
  ko'payadi, 2 soat sig'maydi. Har blokda matn ortig'i bilan olinadi, ball tushgan joy topiladi,
  oxirgi ishonchli gap chegarasidan keyingi blok boshlanadi.
- Kesh (`_kesh_align_*.json`, `_kesh_asr_*.jsonl`) - to'xtagan joydan davom etadi. Birinchi
  urinishda 880/951 segmentdan keyin kompyuter o'chib, hammasi yo'qolgan edi.
- Eksportda bosh va oxirga bir xil 100 ms pad.

**Ochiq savollar:**
- Kotib tezligi bu safar 3.1 s/segment (10 daqiqalik sinovda 1.4 edi). VRAM 7.7-7.9 GB ga
  to'lgan edi, sababi aniqlanmagan.
- Audio hali butunlay xotiraga yuklanadi. 2 soat sig'di (16 kHz mono ~0.5 GB), 20 soat uchun
  bo'laklab yuklash kerak bo'lishi mumkin - sinalmagan.
- PDF'dan qolgan defislar (`quvontirdingiz-lar`) matnda qoladi.

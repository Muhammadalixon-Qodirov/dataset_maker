# Tadqiqot 4: O'zbek tili va kam resursli tillar xususiyatlari

**Doira:** faqat o'zbek tiliga va kam resursli tillarga xos masalalar. Umumiy forced-alignment vositalarini taqqoslash, korpus tarixi va audio sifat chegaralari boshqa agentlar zimmasida.

**Sana:** 2026-09-29. Barcha raqamlar manba URL bilan berilgan. Manbalar qarama-qarshi bo'lsa, ikkisi ham ko'rsatilgan.

---

## 0. Eng muhim xulosalar (qisqa)

1. **MMS-TTS o'zbek modeli faqat KIRILL yozuvida** — `facebook/mms-tts-uzb-script_cyrillic`. Lotin varianti **mavjud emas**. Ishlab chiquvchining stackida bu eng katta arxitektura muammosi.
2. **2026-yil alifbo islohoti amalda** — Senat 2026-09-10 da tasdiqladi: `Oʻ→Ö`, `Gʻ→Ğ`, `Sh→Ş`, `Ch→Ç`. Bu digraf muammosini hal qiladi, lekin butun matn quvuriga uchinchi yozuvni qo'shadi.
3. **Piper uchun o'zbek ovozi mavjud emas.** XTTS-v2 o'zbek tilini **qo'llab-quvvatlamaydi**. NeMo text-processing o'zbek tilini **qo'llab-quvvatlamaydi**. VoxPopuli'da o'zbek **yo'q**. CharsiuG2P'ning o'zbek lug'ati **noto'g'ri** (mualliflar o'zlari ogohlantiradi).
4. **Mavjud bo'lgan narsalar:** espeak-ng `uz` (status: testing), epitran `uzb-Latn` + `uzb-Cyrl`, IS2AI TurkicTTS'ning o'zbek→IPA mapping'i, `uztextnorm` (raqam→so'z), `uztext` (apostrof + 3 yozuv), FeruzaSpeech (60 soat, bitta ayol spiker — TTS uchun tayyor), kuchli o'zbek ASR modellari (CER ~5-7%).
5. **Ishlab chiquvchi o'zi qurishi kerak:** sana/vaqt/telefon raqami normalizatsiyasi, qisqartma kengaytirish, ruscha/inglizcha loan-word talaffuzi, gap tanlash (greedy coverage) skripti, o'zbek fonema inventarizatsiyasi asosida qoplama hisoblash.

---

## 1. Mavjud o'zbek nutq datasetlari va modellari

### 1.1 To'liq jadval — nutq korpuslari

| Resurs | Soat | Spiker | Yozuv | Litsenziya | Havola |
|---|---|---|---|---|---|
| **USC (Uzbek Speech Corpus)** | **105 soat** | 958 | ko'rsatilmagan | **QARAMA-QARSHI:** CC BY 4.0 (ISSAI) / MIT (HF kartasi) | [arXiv 2107.14419](https://arxiv.org/abs/2107.14419), [HF issai/Uzbek_Speech_Corpus](https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus), [GitHub IS2AI/Uzbek_ASR](https://github.com/IS2AI/Uzbek_ASR) |
| **Common Voice uz** (cv-corpus-25.0-2026-03-09) | **265.75 soat yozilgan / 101.01 soat validated** | 2,299 | Lotin | CC-0 | [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmn29fa4y0150o1073fqww73p) |
| **FeruzaSpeech** | **59.61 soat** (kitob 21.57 + BBC Uzbek 38.04) | **1** (Toshkentlik ayol) | **Lotin VA Kirill — ikkisi ham** | CC BY 4.0 | [arXiv 2410.00035](https://arxiv.org/abs/2410.00035), [HF k2speech/FeruzaSpeech](https://huggingface.co/datasets/k2speech/FeruzaSpeech) |
| **UzbekVoice** (uzbekvoice.ai / Mohir.ai) | **~900 soat** (NVIDIA model kartasi bo'yicha) | ko'p (crowdsourced) | Lotin | ko'rsatilmagan | [HF DavronSherbaev/uzbekvoice](https://huggingface.co/datasets/DavronSherbaev/uzbekvoice) (868k qator), [corpus.uzbekvoice.ai](https://corpus.uzbekvoice.ai/en-US/resus) |
| **uzbekvoice-filtered** | 503k qator (501k train / 2.05k val) | ko'p | — | **Apache 2.0** | [HF ai4uz/uzbekvoice-filtered](https://huggingface.co/datasets/ai4uz/uzbekvoice-filtered) |
| **FLEURS uz_uz** | **~12 soat** (har til uchun ~12 soat nutq nazorati) | ko'p | Lotin | CC BY 4.0 | [HF google/fleurs](https://huggingface.co/datasets/google/fleurs), [arXiv 2205.12446](https://ar5iv.labs.arxiv.org/html/2205.12446) |
| **MMS-lab uzb** (Yangi Ahd o'qishlari) | **~32 soat o'rtacha har til uchun** | 1 (ko'pincha erkak) | ko'rsatilmagan | MMS-TTS modeli: **CC-BY-NC-4.0** | [arXiv 2305.13516](https://arxiv.org/pdf/2305.13516), [JMLR 25 (2024)](https://jmlr.org/papers/volume25/23-1318/23-1318.pdf) |
| **CMU Wilderness** | **~20 soat o'rtacha har til uchun** (Yangi Ahd, bible.is) | 1 | — | audio to'g'ridan-to'g'ri tarqatilmaydi; skript bilan qurish kerak | [festvox.org/cmu_wilderness](http://festvox.org/cmu_wilderness/), [GitHub festvox/datasets-CMU_Wilderness](https://github.com/festvox/datasets-CMU_Wilderness) |
| **islomov news YouTube** | 20,795 klip, 16kHz | ko'p | Lotin | Apache 2.0 | [HF islomov/news_youtube_uzbek_speech_dataset](https://huggingface.co/datasets/islomov/news_youtube_uzbek_speech_dataset) — **Gemini 2.5 Pro transkripsiyasi** |
| **islomov IT YouTube** | — | ko'p | Lotin | Apache 2.0 | [HF islomov/it_youtube_uzbek_speech_dataset](https://huggingface.co/datasets/islomov/it_youtube_uzbek_speech_dataset) |
| **islomov Toshkent dialekt podcast** | — | ko'p | Lotin | Apache 2.0 | [HF islomov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset](https://huggingface.co/datasets/islomov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset) |
| **Abduqayum Uzbek-STT-780h** | **~780 soat**, 122,464 misol | ko'p | Lotin | "other" (research/edu) | [HF Abduqayum/Uzbek-STT-Dataset-780h](https://huggingface.co/datasets/Abduqayum/Uzbek-STT-Dataset-780h) — **Gemini avtomatik transkripsiya, so'z darajasida xatolar bor** |
| **openbank-uz YouTube** | 300k qator | ko'p | Lotin | — | [HF openbank-uz/youtube_transcriptions](https://huggingface.co/datasets/openbank-uz/youtube_transcriptions) — Gemini |
| **OvozifyLabs eval** | — (Telegram xabarlari) | ko'p | Lotin | — | [HF OvozifyLabs/asr_evaluate_set](https://huggingface.co/datasets/OvozifyLabs/asr_evaluate_set) |
| **Biruniy** | **200+ soat** (podcast/talk-show), 100+ spiker, 10 dialekt, 99.2% transkript aniqligi | 100+ | Lotin | **TIJORIY** (Research License / Enterprise) | [biruniy.uz](https://www.biruniy.uz/) |
| **VoxPopuli** | **O'ZBEK UCHUN MAVJUD EMAS** — 23 ta EI tili, Yevropa Parlamenti yozuvlari | — | — | — | [arXiv 2101.00390](https://ar5iv.labs.arxiv.org/html/2101.00390) |

**Muhim ogohlantirishlar:**

- **USC ~900 soat EMAS** — 105 soat. 958 raqami **spikerlar** soni. "~900 soat" raqami aslida **UzbekVoice**'ga tegishli (NVIDIA model kartasi: "UzbekVoice: ~900 hrs").
- USC litsenziyasi **qarama-qarshi**: ISSAI e'loni va arXiv sahifasida CC BY 4.0, HF dataset kartasida MIT. Tijoriy foydalanishdan oldin ISSAI bilan aniqlashtirish kerak.
- **Common Voice endi HuggingFace'da yo'q.** 2025-yil oktabrdan Mozilla Common Voice datasetlari faqat Mozilla Data Collective orqali tarqatiladi; `mozilla-foundation/common_voice_17_0` HF kartasi bo'sh.
- Gemini bilan transkripsiya qilingan datasetlar (islomov, Abduqayum, openbank-uz) **TTS uchun yaramaydi** — label shovqini bor, uni mualliflar o'zlari tan oladi. Ular ASR fine-tune uchun mos.

### 1.2 TTS modellari — o'zbek tili uchun

| Model | Arxitektura | Yozuv | Litsenziya | Natija | Havola |
|---|---|---|---|---|---|
| **facebook/mms-tts-uzb-script_cyrillic** | VITS, 36.3M param, 16kHz | **FAQAT KIRILL** | **CC-BY-NC-4.0** (nokommersiya!) | — | [HF](https://huggingface.co/facebook/mms-tts-uzb-script_cyrillic) |
| `mms-tts-uzb-script_latin` / `mms-tts-uzn` | — | — | — | **MAVJUD EMAS** | qidiruvda topilmadi |
| **IS2AI/TurkicTTS** | Tacotron 2 + WaveGAN, qozoq tilida o'qitilgan, zero-shot | **O'zbek Lotin** (`Oʻzbekiston` — U+02BB bilan) | ko'rsatilmagan | **MOS 2.85 / tushunarlilik 80% / aniqlik (SUS) 45%** (22 baholovchi) | [GitHub IS2AI/TurkicTTS](https://github.com/IS2AI/TurkicTTS), [Interspeech 2023](https://www.isca-archive.org/interspeech_2023/yeshpanov23_interspeech.pdf) |
| **Piper (rhasspy)** | VITS ONNX | — | — | **O'ZBEK OVOZI MAVJUD EMAS** — VOICES.md'da uz yo'q. Qozoq (kk_KZ: iseke, issai, raya) va turk (tr_TR: dfki, fahrettin, fettah) bor | [piper VOICES.md](https://github.com/rhasspy/piper/blob/master/VOICES.md) |
| **XTTS-v2 (Coqui)** | — | — | — | **O'ZBEK QO'LLAB-QUVVATLANMAYDI.** 17 til: ar, zh-cn, cs, nl, en, fr, de, hi, hu, it, ja, ko, pl, pt, ru, es, **tr** | [HF coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) |
| **uzlm/sayro-tts-1.7B** | Qwen3-TTS-12Hz-1.7B-Base fine-tune | ko'rsatilmagan | maxsus "sayro-terms-of-use", **gated** (qo'lda tasdiqlash) | MOS/CER **e'lon qilinmagan** | [HF uzlm/sayro-tts-1.7B](https://huggingface.co/uzlm/sayro-tts-1.7B) |
| **AbrorBalxiyev/uzbek-tts-model** | SpeechT5 fine-tune (audiofolder) | — | — | baholanmagan | [HF](https://huggingface.co/AbrorBalxiyev/uzbek-tts-model) |

**Eski akademik ish:** 31 soat o'zbek nutqi bilan Tacotron + Parallel WaveGAN, **MOS 4.36/5** — ["Uzbek Speech Synthesis Using Deep Learning Algorithms"](https://link.springer.com/chapter/10.1007/978-3-031-27199-1_5). Dataset va kod **ommaviy emas**.

TurkicTTS maqolasi o'zi tan oladi: o'zbek TTS bo'yicha faqat konkatenativ (unit-selection) ishlar topilgan, ularning datasetlari "mavjud emas yoki zamonaviy TTS arxitekturalari uchun yaroqsiz" ([Interspeech 2023, 2-bo'lim](https://www.isca-archive.org/interspeech_2023/yeshpanov23_interspeech.pdf)).

### 1.3 CER back-check uchun ASR modellari

**Whisper zero-shot o'zbek tilida yaramaydi.** Whisper maqolasidagi eng kichik o'zbek WER — **90.2%** (BlueRaccoon model kartasida keltirilgan: ["The smallest reported WER of 90.2 for the Uzbek language" in the original Whisper paper](https://huggingface.co/BlueRaccoon/whisper-small-uz)). Ya'ni o'zbek Whisper'ning 99 tilida bor, lekin fine-tune qilinmagan holda foydasiz.

Fine-tune qilingan modellar — **alphacep/awesome-speech** benchmark jadvali ([manba](https://github.com/alphacep/awesome-speech/blob/main/uzbek.md)), WER %:

| Model | CV | Digits | Fleurs | ISSAI | ISSAI Rare | Feruza | Ovozify |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vosk Small 0.24 | 13.59 | 42.42 | 27.34 | 13.94 | 24.19 | 17.99 | 57.81 |
| Vosk 0.60 Streaming | 9.74 | 25.80 | 17.67 | 15.42 | 35.11 | 7.71 | 48.34 |
| MMS1B + LM | 17.24 | 80.06 | 23.39 | 28.18 | 43.70 | – | – |
| Omnilingual LLM 7B V2 | 20.74 | 57.92 | 16.90 | 29.30 | 45.44 | – | – |
| Rifkat STT v2 (wav2vec+LM) | 6.46 | 28.02 | 20.03 | 7.74 | 21.75 | 8.89 | 44.44 |
| Nvidia FastConformer | 8.86 | 57.49 | 15.10 | 14.37 | 28.85 | 6.61 | 54.56 |
| Nvidia FastConformer Uzinfocom | 9.18 | 41.93 | 14.10 | 14.07 | 30.45 | 6.71 | 36.02 |
| **Whisper Medium Kotib** | **8.92** | **9.74** | **7.72** | **14.05** | **33.42** | **5.93** | **22.04** |
| Whisper Medium Rubai | 8.52 | 10.03 | 4.74* | 12.06 | 31.73 | 6.65 | 24.58 |
| Whisper Medium Navai | 7.28 | 25.95 | 7.64 | 9.21 | 24.42 | 7.64 | 46.06 |
| GigaAM Multilingual | 7.24 | 46.79 | 11.96 | 11.30 | 29.58 | 6.58 | 23.17 |
| **GigaAM Multilingual Large** | **5.52** | 38.95 | 8.82 | 8.77 | 26.40 | 5.93 | 20.82 |
| **ROVER (ansambl)** | **4.64** | **11.85** | **5.57** | **7.23** | **22.27** | **5.15** | **19.88** |

\* Rubai'ning Fleurs natijasi ishonchsiz — test to'plami o'qitishga kirgan bo'lishi mumkin (manba o'zi ogohlantiradi).

**Amaliy xulosa CER back-check uchun:** eng yaxshi ochiq modellar o'qib berilgan toza nutqda (Feruza ustuni) **5.15–6.6% WER** beradi. Ya'ni CER back-check chegarasini belgilashda ASR'ning o'zi ~5% WER shovqin kiritadi. Buni dataset xatosi deb hisoblash mumkin emas.

**Muhim modellar:**

- **Kotib/uzbek_stt_v1** (ishlab chiquvchining o'z stacki): Whisper Medium (769M), **~1,600 soat** o'qitilgan (foundation 725s + robustness 394s + domain adaptation 474s), **umumiy WER 16.7%, CER 7.0%**, toza nutq ~6-11% WER, shovqinli 12-24%, dialektlar 16-25%. **Lotin yozuvi**, ruscha loan-word'larni lotin harflarida transkripsiya qiladi. **Apache 2.0**. [HF Kotib/uzbek_stt_v1](https://huggingface.co/Kotib/uzbek_stt_v1)
- **nvidia/stt_uz_fastconformer_hybrid_large_pc**: 1,000 soat (CV 17.0 uz ~90s + UzbekVoice ~900s + FLEURS uz ~10s), ~115M param, **CC-BY-4.0**, punktuatsiya+bosh harf bilan o'qitilgan. FLEURS DEV 17.52% (punktuatsiyasiz 12.20%), MCV TEST 16.46% (punktuatsiyasiz 7.78%). **Lotin va Kirill ikkisini ham qo'llaydi.** 16kHz mono. [HF](https://huggingface.co/nvidia/stt_uz_fastconformer_hybrid_large_pc)
- **MMS-ASR uzb** (`facebook/mms-1b-all`): mavjud, lekin o'zbek tilida zaif — **CV 17.24% WER** (LM bilan), Digits 80.06%. CER back-check uchun tavsiya etilmaydi.
- **Seamless**: o'zbek uchun alohida benchmark topilmadi.

---

## 2. O'zbek yozuvi muammosi

### 2.1 Lotin / Kirill va 2021–2026 islohoti

O'zbek tili **uchta** faol yozuvda ishlatiladi (Afg'onistondagi Janubiy o'zbek arab yozuvini hisobga olmaganda):

1. **Kirill** (1940-1993) — hali ham keng qo'llanadi, ayniqsa katta yoshdagilar orasida
2. **Hozirgi Lotin** (1993 qonuni, 1995 tuzatishi) — 26 harf + 3 digraf (`sh`, `ch`, `ng`) + `Oʻ`, `Gʻ`
3. **Yangi Lotin (2026)** — 28 harf + 1 apostrof

Manbalar: [Wikipedia: Uzbek alphabet](https://en.wikipedia.org/wiki/Uzbek_alphabet); 1993-yil qonuni [lex.uz/docs/-112286](https://lex.uz/docs/-112286), 1995 tuzatishi [lex.uz/docs/-116158](https://lex.uz/docs/-116158).

**2021-yil rejasi:** Vazirlar Mahkamasining 2021-02-10 dagi 61-son qarori ([lex.uz/uz/docs/-5281850](https://lex.uz/uz/docs/-5281850)) barcha huquqiy matnlarni **2023-yilga qadar** yangi lotin alifbosiga o'tkazishni belgilagan edi. Bu **amalga oshmadi**.

**2026-yil islohoti (HOZIRGI HOLAT):**

- Quyi palata qonunni **2026-07-09** da qabul qildi ([gazeta.uz](https://www.gazeta.uz/en/2026/07/09/alphabet/))
- Senat **2026-09-10** da 19-yalpi majlisda tasdiqladi va Prezidentga imzolashga yubordi ([gazeta.uz](https://www.gazeta.uz/en/2026/09/10/uzb-alphabet/), [Times of Central Asia](https://timesca.com/uzbekistan-latin-alphabet-changes-senate/))
- 2026-09-29 holatiga ko'ra Prezident imzolagani haqida tasdiq topilmadi

**O'zgarishlar:**

| Hozirgi | Yangi (2026) |
|---|---|
| `Oʻ oʻ` | **`Ö ö`** |
| `Gʻ gʻ` | **`Ğ ğ`** |
| `Sh sh` | **`Ş ş`** |
| `Ch ch` | **`Ç ç`** |
| `Ng ng` | alifbodan **olib tashlandi** (imlo qoidalarida saqlanadi) |

Jami: **26 harf + 3 harf birikmasi → 28 harf + 1 apostrof**. Senator Odiljon Mamatkarimov asosni "bir tovush — bir harf" printsipi deb izohladi ([gazeta.uz](https://www.gazeta.uz/en/2026/09/10/uzb-alphabet/)).

**O'tish davri:** hozirgi alifboda chiqarilgan hujjatlar, milliy valyuta va qimmatli qog'ozlar aylanishda qoladi; ko'rsatkichlar, geografik nomlar belgilari va bino elementlari eskirgunicha yoki qonun amalga oshirish programmasidagi muddatlargacha ishlatilishi mumkin. Aniq muddatlar e'lon qilinmagan.

### 2.2 Modellar qaysi yozuvni kutadi — AMALIY JADVAL

| Komponent | Kutadigan yozuv | Manba |
|---|---|---|
| **MMS-TTS uzb** | **KIRILL** (model nomi: `mms-tts-uzb-script_cyrillic`) | [HF](https://huggingface.co/facebook/mms-tts-uzb-script_cyrillic) |
| **espeak-ng `uz`** | Rasmiy hujjatda "With Cyrillic alphabet", LEKIN `uz_rules` faylida `.replace` bo'limi Kirill→Lotin o'giradi, keyin Lotin qoidalari (`oʻ`, `gʻ`, `sh`→`S`, `ch`→`tS`) ishlaydi → **amalda ikkisini ham qabul qiladi** | [languages.md](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md), [dictsource/uz_rules](https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/dictsource/uz_rules) |
| **epitran** | **ikkisi ham**: `uzb-Latn` va `uzb-Cyrl` | [epitran README](https://github.com/dmort27/epitran) |
| **IS2AI TurkicTTS** | **Lotin** (`Oʻzbekiston` — U+02BB bilan) | [GitHub](https://github.com/IS2AI/TurkicTTS) |
| **Kotib STT** | **Lotin** | [HF](https://huggingface.co/Kotib/uzbek_stt_v1) |
| **NVIDIA FastConformer uz** | ikkisi ham | [HF](https://huggingface.co/nvidia/stt_uz_fastconformer_hybrid_large_pc) |
| **FeruzaSpeech** | ikkisi ham beriladi | [arXiv 2410.00035](https://arxiv.org/html/2410.00035v1) |

**Bu ishlab chiquvchi uchun nima anglatadi:** agar MMS-TTS'ni baza sifatida ishlatmoqchi bo'lsa, Lotin matnini **Kirillga o'girish** kerak. Lekin Lotin→Kirill transliteratsiya **eng zaif yo'nalish** (pastda 2.4-bo'limga qarang) — har 10 so'zdan biri xato bo'lishi mumkin. Alternativa: MMS-TTS tokenizer'ining vocab'ini Lotin belgilariga qayta moslash (embedding layer'ni almashtirish) — bu fine-tune vaqtida qilinadigan ish.

### 2.3 Apostrof muammosi — nechta variant bor

**Rasmiy to'g'ri belgilar:**

| Belgi | Unicode | Vazifasi |
|---|---|---|
| `ʻ` | **U+02BB** MODIFIER LETTER TURNED COMMA | `oʻ` va `gʻ` harflari uchun |
| `ʼ` | **U+02BC** MODIFIER LETTER APOSTROPHE | tutuq belgisi (glottal to'xtam), masalan `sanʼat` |

Manba: [Wikipedia: Uzbek alphabet](https://en.wikipedia.org/wiki/Uzbek_alphabet), [Wikipedia: Oʻ](https://en.wikipedia.org/wiki/O%CA%BB), [Wikipedia: Gʻ](https://en.wikipedia.org/wiki/G%CA%BB).

**Nega buzilib ketadi:** U+02BB Microsoft Windows'dagi o'zbek lotin klaviatura maketlarida (2022 holati) **yozib bo'lmaydi**, shuning uchun boshqa belgilar bilan almashtirish juda keng tarqalgan ([Wikipedia: Oʻ](https://en.wikipedia.org/wiki/O%CA%BB)). U+2018 va U+2019 o'zbek lotinida **ikkilamchi qo'shtirnoq** rolini bajaradi — ya'ni ularni harf o'rnida ishlatish semantik jihatdan ham noto'g'ri.

**Haqiqiy matnda uchraydigan variantlar** — bu maydonni aniq raqamlagan ikki manba:

1. **`uztext` kutubxonasi**: "**twelve look-alikes**" — 12 ta o'xshash belgini to'g'ri Unicode'ga xaritalaydi (U+02BB `oʻ`/`gʻ` uchun, U+02BC tutuq belgisi uchun) ([GitHub abafaboy/uztext](https://github.com/abafaboy/uztext))
2. **Turkic script unification tadqiqoti**: preprocessing'da quyidagi ro'yxatni ASCII apostrofga (U+0027) normallashtiradi — **U+02BB, U+2018, U+2019, U+201B, U+0060, U+00B4, U+02BC, U+FF07** ([arXiv 2608.09356](https://arxiv.org/pdf/2608.09356))
3. **UzTransliterator maqolasi** ham tan oladi: "there is a big deal of inconsistency in writing o' and g' letters in the currently official alphabet due to the use of apostrophe, which comes in many ways, such as o\`, o\`, o', and g\`, g\`, g' forms respectively" ([CEUR Vol-3315 paper05](https://ceur-ws.org/Vol-3315/paper05.pdf))

**Odamlar qanday normallashtiradi — ikki qarama-qarshi yondashuv:**

- **Kanonlashtirish** (`uztext`): hamma variantni **U+02BB**'ga yig'ish — `o'zbek`, `o‘zbek`, `o`zbek` → `oʻzbek`
- **ASCII'ga tushirish** (Turkic NLP tadqiqoti): hammasini **U+0027**'ga tushirish — model vocab'ini kichraytirish uchun
- **Butunlay yo'q qilish**: 2026 alifbosiga o'tish — `Ö`/`Ğ` apostrof muammosini butunlay hal qiladi

**TTS uchun tavsiya:** dataset yozuvida **bitta** kanonik shaklni tanlang va uni matn quvurining eng boshiga qo'ying. Bu tanlov TTS model vocab'ining o'lchamiga to'g'ridan-to'g'ri ta'sir qiladi.

**Homoglif ogohlantirishi:** TurkicTTS mualliflari alohida ta'kidlaydi — "we observed the interchangeable use of visually similar/identical characters from different scripts — also known as homoglyphs — in almost all Turkic languages that have transitioned from the Cyrillic to the Latin alphabet. Normally, this does not bother the reader, but can be problematic for computer systems. Therefore, the collected text should be carefully inspected." ([Interspeech 2023](https://www.isca-archive.org/interspeech_2023/yeshpanov23_interspeech.pdf)). Ya'ni Kirill `с`, `а`, `о`, `е`, `р`, `х` harflari lotin matni ichiga aralashib ketadi.

### 2.4 Transliteratsiya kutubxonalari va ularning aniqligi

| Kutubxona | Yo'nalishlar | Aniqlik | Litsenziya | Oxirgi yangilanish | Havola |
|---|---|---|---|---|---|
| **UzTransliterator** | Kirill ↔ Lotin ↔ Yangi Lotin | **micro-F1 0.89–0.94** so'z darajasida (~9,600 so'z, bosma imlo lug'atidan) | **ko'rsatilmagan** | **0.0.36, 2022-10-23 — eskirgan** | [GitHub](https://github.com/UlugbekSalaev/UzTransliterator), [PyPI](https://pypi.org/project/UzTransliterator/), [CEUR Vol-3315](https://ceur-ws.org/Vol-3315/paper05.pdf), [arXiv 2205.09578](https://arxiv.org/abs/2205.09578) |
| **uztext** | Kirill ↔ Lotin ↔ Yangi Lotin (2026); `to_latin`, `to_cyrillic`, `to_new`, `from_new`, `normalize`, `search_key`, `sum_words` | 699 test keysi (`corpus/cases.json`), har qoida uchun manba (RULES.md) | **MIT** | 2026 (aktiv) | [GitHub abafaboy/uztext](https://github.com/abafaboy/uztext) — Python 3.10+ va TypeScript/npm |
| **uztextnorm** | Kirill → Lotin (bir yo'nalishli) | — | **MIT** | **0.5.3, 2026-08-26** | [PyPI](https://pypi.org/project/uztextnorm/), [GitLab adizbek/uztextnorm](https://gitlab.com/adizbek/uztextnorm) |
| **uzbek-latin-cyrillic-converter** | Lotin ↔ Kirill | baholanmagan | — | — | [PyPI](https://pypi.org/project/uzbek-latin-cyrillic-converter/) |

**UzTransliterator aniqlik jadvali** (micro-averaged F1, so'z darajasi, ~9,600 so'zli baholash):

- **Lotin → Yangi Lotin: 0.94** (eng yaxshi — faqat 5 harf o'zgaradi, istisnolar yo'q; qolgan xatolar faqat qisqartmalar bilan)
- **Lotin → Kirill: 0.89** (eng yomon — 11 qoida + ko'p istisnolar)
- Qolgan yo'nalishlar 0.90–0.93 oralig'ida

Manba: [CEUR Vol-3315 paper05, Table 3 va 4-bo'lim](https://ceur-ws.org/Vol-3315/paper05.pdf).

**Ma'lum xato holatlari (maqola o'zi sanab o'tadi):**

1. **Bosh harfli digraf muammosi**: Kirill `Ш` va `Ю` → Lotin. `SH`/`YU` qilinsa, `SHoʻrva` (sho'rva), `YUlduz` (yulduz) kabi xato; `Sh`/`Yu` qilinsa, akronimlarda xato: `AQSh` (AQSH), `YuNESKO` (YUNESKO). Yechim: atrofdagi harflarni hisobga olish.
2. **Tutuq belgisi (glottal to'xtam) tushib qolishi**: Kirill shaklida bor, Lotinda yo'q — `фaкультет` → `fakultet`, `кaльций` → `kalsiy`. Teskari yo'nalishda glottal to'xtam "yo'qdan paydo bo'lishi" kerak. Yechim: istisno so'zlar ro'yxati.
3. **Ruscha loan-word'lar**: istisno so'zlar bazasida **233 so'z** bor, ularning **ko'pchiligi ruscha o'zlashmalar**. Kirill `ц`, `ъ` (glottal), `щ` harflari muammo tug'diradi. Misollar: `aksent`, `budilnik`, `batalyon`, `feldsher`, `fransuz`, `intervyu`, `koeffitsiyent`, `korrupsiya`, `kuryer`, `medalyon`, `oktabr`, `pavilyon`, `porshen`, `shpatel`, `cherepitsa`.
4. **Agglyutinatsiya**: istisno bazasida faqat **lemmalar** saqlanadi, lekin o'zbek tilida so'zlar asosan qo'shimchali shakllarda uchraydi → o'zak shakllarni saqlash yoki sintaktik bilim qo'shish kerak.
5. **`Ts`/`S`** va **`E`/`Ye`** xaritalanishi bir-biriga bir xil emas (Table 2'da "murakkab xaritalash" deb belgilangan).

**Xulosa:** UzTransliterator "state-of-the-art" deb atalsa-da, 2022-yildan beri yangilanmagan va litsenziyasi ko'rsatilmagan. **`uztext` (MIT, 2026)** hozirgi eng yaxshi tanlov — 2026 alifbosini ham qo'llaydi.

### 2.5 Digraflar (sh, ch, ng) — CTC alignment va G2P uchun oqibatlari

**Faktlar:**

- Hozirgi rasmiy alifboda `sh`, `ch`, `ng` — **ikki lotin belgisi bitta fonemani** ifodalaydi: `sh` → /ʃ/ (espeak'da `S`), `ch` → /tʃ/ (espeak'da `tS`), `ng` → /ŋ/ (espeak'da `N`) ([espeak-ng uz_rules](https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/dictsource/uz_rules))
- `Oʻ` va `Gʻ` ham **harf + modifikator** = 2 kod nuqtasi, 1 fonema
- 2026 islohoti aynan shuni tuzatadi — "bir tovush — bir harf" ([gazeta.uz](https://www.gazeta.uz/en/2026/09/10/uzb-alphabet/))

**Belgi darajasidagi CTC alignment'ga ta'siri:**

Belgi darajasida ishlaydigan CTC modeli `sh` uchun **ikki** chiqish tokenini bashorat qilishi kerak, ammo audioda **bitta** fonema turadi. Buning natijasi:

- `s` va `h` tokenlari o'rtasida sun'iy chegara paydo bo'ladi — vaqt belgilari (timestamps) fonema chegarasiga to'g'ri kelmaydi
- `ng` eng xavfli: `n`+`g` ketma-ketligi ham /ŋ/ (bitta fonema, masalan `keng`), ham /n/+/g/ (ikki fonema, masalan morfema chegarasida `son+ga`) bo'lishi mumkin. Buni **faqat orfografiyadan aniqlab bo'lmaydi** — morfologik bilim kerak
- `sh` ham ikki xil bo'lishi mumkin: /ʃ/ (`shakar`) yoki /s/+/h/ (morfema chegarasida, masalan `ishxona` kabi holatlar — `s`+`h` emas, lekin prefiks/suffiks chegaralarida uchraydi)

**Praktik yechimlar (ishlab chiquvchi tanlashi kerak):**

1. **Digraflarni bitta belgiga siqish** — aynan 2026 alifbosi (`ş`, `ç`, `ö`, `ğ`) qiladigan ish. `uztextnorm`'da `script="2026"` opsiyasi mavjud. Bu CTC vocab'ni ham kichraytiradi.
2. **Kirillga o'girish** — Kirillda `ш`, `ч`, `ў`, `ғ` allaqachon bitta belgi. Lekin `нг` Kirillda ham ikki belgi.
3. **Fonema darajasida ishlash** — espeak-ng yoki epitran orqali IPA'ga o'tish, keyin fonema darajasida alignment qilish.

**Muhim:** bu bo'lim faktlarga (digraf tuzilishi, espeak qoidalari, islohot asosi) tayanadi; CTC alignment'ga aniq ta'siri o'zbek tili uchun alohida o'lchanmagan — bunday tadqiqot topilmadi.

---

## 3. O'zbek matnini TTS uchun normalizatsiya qilish

### 3.1 Mavjud kutubxonalar

#### `uztextnorm` — asosiy topilma

**Versiya 0.5.3, MIT litsenziya, 2026-08-26 da chiqarilgan, Python 3.10+.** [PyPI](https://pypi.org/project/uztextnorm/) · [GitLab adizbek/uztextnorm](https://gitlab.com/adizbek/uztextnorm)

O'zini "Uzbek Text Normalization Library for **TTS** and NLP applications" deb ta'riflaydi.

**QILADIGAN ishlari:**

| Vazifa | Misol |
|---|---|
| Unicode NFC normalizatsiya | — |
| Kirill → Lotin transliteratsiya | `Ўзбекистон Республикаси` → `o'zbekiston respublikasi` |
| **Butun sonlar** | `123` → `bir yuz yigirma uch` |
| **O'nli kasrlar** | `3.14` → `uch butun o'n to'rt` |
| **Oddiy kasrlar** | `3/4` → `to'rtdan uch`; `1/2` → `yarim`; `2 3/4` → `ikki butun to'rtdan uch` |
| **Valyuta** | `10,000 so'm` → `o'n ming soʻm`; `2,500 $` → `ikki ming besh yuz dollar` |
| **Foiz** | `15%` → `o'n besh foiz` |
| **Rim raqamlari** | `XXI asr` → `yigirma birinchi asr` (`expand_roman=True`) |
| **Qolgan raqamlarni harflab o'qish** | `A350` → `a uch yuz ellik` (`spell_leftover_digits=True`) |
| URL/email himoyasi | `speak_urls=True` bo'lsa `https://kun.uz` → `kun nuqta uz` |
| Punktuatsiya normalizatsiyasi | — |
| **2026 alifbosi** | `script="legacy"` (o'/g'/sh/ch) yoki `script="2026"` (ö/ğ/ş/ç) |
| Maxsus qoidalar | `custom_pre_rules` / `custom_post_rules` |

**QILMAYDIGAN ishlari — ishlab chiquvchi o'zi qurishi kerak:**

- ❌ **Sanalar** (`2026-09-29`, `10-sentabr`, `1995 yil 6 may`)
- ❌ **Vaqt** (`14:30`, `soat 9:15`)
- ❌ **Telefon raqamlari** (`+998 90 123 45 67`)
- ❌ **Qisqartmalar / akronimlar** (`AQSh`, `BMT`, `MDH`, `O'zR`, `v.b.`, `va h.k.`, `t.y.`)
- ❌ **Tartib sonlar alohida** (`5-sinf`, `3-uy`, `2-kurs` — o'zbekchada `-chi`/`-inchi` qo'shimchasi bilan)
- ❌ **Inglizcha/ruscha so'zlarning talaffuzi** (matn ichida lotin yozuvida qolgan `Microsoft`, `WhatsApp`, `Instagram`)

#### `uztext` — apostrof + 3 yozuv + summa

**MIT, 2026, 699 test keysi.** [GitHub abafaboy/uztext](https://github.com/abafaboy/uztext)

- `to_latin`, `to_cyrillic`, `to_new`, `from_new`, `normalize`, `search_key`, `sum_words`
- 12 ta apostrof o'xshashini U+02BB / U+02BC'ga xaritalaydi
- `sum_words`: `1 250 000,50` → `bir million ikki yuz ellik ming soʻm ellik tiyin` (valyuta va yozuv opsiyalari bilan)
- Python (0 dependency, CLI bor) va TypeScript/npm — bitta umumiy test korpusi
- Har qoida uchun manba RULES.md'da hujjatlashtirilgan (rasmiy lug'at va imlo standartlari)

#### `NeMo-text-processing` — **O'ZBEK UCHUN MAVJUD EMAS**

NVIDIA WFST-asosidagi TN/ITN kutubxonasi quyidagi tillarni qo'llaydi: en, de, es, fr, hu, sv, zh, ar, it, hy, ja, hi, ko, vi, pt. **O'zbek (uz) yo'q.** Grammatikani qo'lda yozib qo'shish mumkin, lekin bu katta ish. Manba: [NVIDIA/NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing), [NeMo TN docs](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/nlp/text_normalization/wfst/wfst_text_normalization.html).

### 3.2 Boshqa o'zbek NLP vositalari (kontekst uchun)

- **uzbek-tagger-bert** — Tahrirchi-BERT asosida POS tagger; o'zi tan oladigan muammolar: agglyutinativ so'z yasalishi, kontekstual omonimlarni ajratish, **apostrof normalizatsiyasi** ([GitHub](https://github.com/MaksudSharipov/uzbek-tagger-bert))
- **UzBERT** ([arXiv 2108.09814](https://arxiv.org/abs/2108.09814)), UzWordNet, o'zbek stop-words dataseti, morfologik analizator (Prolog) — UzTransliterator maqolasida sanab o'tilgan
- **Janubiy o'zbek** (Afg'oniston, arab yozuvi) uchun alohida resurslar: ["Filling the Gap for Uzbek: Creating Translation Resources for Southern Uzbek"](https://arxiv.org/pdf/2508.14586) — bu loyihaga tegishli emas, lekin "o'zbek" deb belgilangan datasetlarni tekshirishda ehtiyot bo'lish kerakligini ko'rsatadi

### 3.3 Unli harmoniya va agglyutinatsiya — fonema qoplamasiga oqibatlari

**MUHIM TUZATISH:** ko'pchilik turkiy tillardan farqli ravishda **o'zbek tilida unli harmoniya deyarli yo'q**.

> "Tashkent Uzbek, and hence also standard Uzbek, exhibit **very limited vowel harmony**." — Shinji Ido, "Uzbek", *Journal of the International Phonetic Association*, 2025 ([Cambridge Core](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/uzbek/48B3BBA8C522A61CEAE0E21193B3C779))

Wikipedia ham tasdiqlaydi: "Unlike all other Turkic languages, Uzbek does not have vowel harmony."

Faqat "deverbalizing suffix" kabi bir nechta qo'shimchada harmoniya qoldiqlari ko'rinadi (Ido 2025).

**Amaliy oqibat:** Turkcha yoki qozoqcha TTS'da qo'shimchalarning old/orqa unli variantlarini alohida modellash kerak. **O'zbek tilida bu kerak emas** — bu ishni soddalashtiradi. Ya'ni qo'shimchalar deyarli o'zgarmas shaklda keladi va fonema qoplamasini hisoblashda alohida variantlarni sanash shart emas.

**Agglyutinatsiya esa muammo bo'lib qoladi:**

- O'zbek tili "highly-agglutinative, null-subject, null-gender" til, Turkiy tillarning **Qarluq** shoxida ([CEUR Vol-3315](https://ceur-ws.org/Vol-3315/paper05.pdf))
- So'z boyligi amalda cheksiz — lug'at asosidagi G2P yaramaydi, **qoida asosidagi** G2P kerak (espeak-ng/epitran aynan shunday ishlaydi)
- Transliteratsiya istisno bazalari faqat lemmalarni saqlaydi → qo'shimchali shakllarda ishlamaydi (2.4-bo'limga qarang)
- **Bosh qo'shma undoshlar yo'q**: o'zbek tilida so'z boshida undosh klasterlar taqiqlangan, aksariyat bo'g'inlar bitta undosh yoki unli bilan boshlanadi (CV- tuzilishi) ([Wikipedia: Uzbek phonology](https://en.wikipedia.org/wiki/Uzbek_alphabet) va [MustGo](https://www.mustgo.com/worldlanguages/uzbek/)) → bo'g'in tuzilishlari soni cheklangan, bu qoplamani osonlashtiradi

---

## 4. Fonema/difon qoplamasi va skript tanlash

### 4.1 O'zbek fonema inventarizatsiyasi

**Asosiy manba:** Shinji Ido, "Uzbek", *Journal of the International Phonetic Association*, **2025** — Toshkent o'zbek shevasi (standart variant) uchun IPA tasviri. [Cambridge Core](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/uzbek/48B3BBA8C522A61CEAE0E21193B3C779)

**Unlilar: 6 ta** — /i, e, ɜ, a, o, u/

(Boshqa manbalar bir oz boshqacha ro'yxat beradi: /a/, /æ/, /i/, /o/, /u/, /ø/ — lekin **soni bir xil: 6**.)

**Undoshlar:** Ido 2025'dan olingan ro'yxat (WebFetch bilan, to'liq bo'lmasligi mumkin — `/l/`, `/z/`, `/ʁ/` ro'yxatda ko'rinmadi, lekin ular o'zbek fonemalari):
/p, b, t, d, k, ɡ, q, ʔ, m, n, ŋ, f, v, s, ɕ, χ, h, ɾ, j, t͡ɕ, d͡ʑ, t͡s/

**Shimoliy o'zbek uchun boshqa manba: 25 undosh fonema** ([Wikipedia, Uzbek phonology bo'limlari orqali](https://en.wikipedia.org/wiki/Uzbek_alphabet); [ResearchGate: Phonemic Consonant Inventory of Uzbek](https://www.researchgate.net/figure/Phonemic-Consonant-Inventory-of-Uzbek_tbl65_287244076))

**Amaliy hisob:** **~6 unli + ~24-25 undosh = ~30-31 fonema.** Xarakterli belgilar: uvulyar/velyar /q/, /ʁ/, /χ/ — boshqa turkiy tillarda ham bor, ammo ingliz tilida yo'q (ya'ni ingliz tilidan transfer learning'da bu fonemalar eng zaif bo'ladi).

**IPA xaritalash resursi:** IS2AI TurkicTTS repositoriysida `ipa_mapping.pdf` va `ipa_convert.py` — o'zbek lotin harflarini IPA'ga xaritalaydi (10 turkiy til uchun, qo'lda tuzilgan). Maqola aytadi: "The mappings ... were **manually created** based on our expertise, as we could not find a complete mapping that would allow an error-free conversion". Ya'ni **bu o'zbek uchun ochiq mavjud eng to'liq harf→IPA jadvali**. ([GitHub IS2AI/TurkicTTS](https://github.com/IS2AI/TurkicTTS), [Interspeech 2023, Table 2](https://www.isca-archive.org/interspeech_2023/yeshpanov23_interspeech.pdf))

### 4.2 O'zbek G2P / fonemizator resurslari — nima bor, nima yo'q

| Vosita | O'zbek holati | Tafsilot |
|---|---|---|
| **espeak-ng `uz`** | ✅ **BOR, lekin `status testing`** | Voice fayli aynan shunday deydi: `name Uzbek / language uz / status testing`. Muallif: Andiv06. `trk` (turkiy) oilasida. `uz_rules` Kirill→Lotin `.replace` qiladi, keyin `oʻ`, `gʻ`, `sh`→`S`, `ch`→`tS`, `ng`→`N` qoidalari ishlaydi. Manbalar: [voice fayli](https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/espeak-ng-data/lang/trk/uz), [uz_rules](https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/dictsource/uz_rules), [languages.md](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md) |
| **epitran** | ✅ **BOR — ikki yozuvda** | `uzb-Latn` va `uzb-Cyrl` qo'llab-quvvatlanadi. [GitHub dmort27/epitran](https://github.com/dmort27/epitran), [Epitran maqolasi (LREC 2018)](https://aclanthology.org/L18-1429.pdf) |
| **CharsiuG2P** | ❌ **ISHLATMANG** | README boshida to'g'ridan-to'g'ri ogohlantirish: *"We found that the pronounciation dictionary for Uzbek `uzb` is **incorrect**. Please **do not use it** in your applicaion."* Mualliflar uni yaxshiroq resurs bilan almashtirmoqchi. [GitHub lingjzhu/CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P) |
| **IS2AI TurkicTTS `ipa_convert.py`** | ✅ **BOR** | O'zbek Lotin → IPA, qo'lda tuzilgan, 42 IPA belgisi (qozoq alifbosi asosida) |
| **transphone** | tekshirilmagan | 8k til uchun fonema tokenizatori; o'zbek sifati baholanmagan ([GitHub xinjli/transphone](https://github.com/xinjli/transphone)) |
| **O'zbek talaffuz lug'ati (CMUdict uslubida)** | ❌ **ochiq mavjud emas** | Qidiruvda topilmadi. CMU Wilderness "word pronunciations" beradi, lekin sifati baholanmagan |

**Xulosa:** o'zbek G2P uchun **ikki ishlaydigan yo'l** bor — espeak-ng (`testing` sifat bilan) va epitran (`uzb-Latn`/`uzb-Cyrl`). Ikkisining aniqligini o'zbek tilida o'lchagan nashr topilmadi, shuning uchun **ishlab chiquvchi o'zi qo'lda tekshirishi kerak** (masalan 200-500 so'zli qo'lda yozilgan talaffuz ro'yxati bilan).

### 4.3 Gap tanlash algoritmlari — professional TTS skriptlari qanday quriladi

**Asosiy algoritm: greedy set-cover.** Ishlash printsipi:

> "The greedy algorithm builds the cover step-by-step by recursively selecting sentences — the first sentence selected has the largest unit count, units in that sentence are removed from the to-be-covered list, and the sentence covering the maximum number of remaining units is selected next, continuing until all units are covered or the corpus is exhausted."

Qoplanadigan birliklar: **fonemalar → difonlar (biphone) → trifonlar (triphone)**, prosodik va fonetik kontekstlar bilan.

**Aniq algoritm nomlari va vositalar:**

| Nom / vosita | Tafsilot | Havola |
|---|---|---|
| **Greedy selection** (klassik) | TTS korpus dizayni uchun standart. Difon/trifon/so'z qoplamasi bo'yicha | [Bozkurt et al., "Text design for TTS speech corpus building using a modified greedy selection"](https://www.researchgate.net/publication/221478056_Text_design_for_TTS_speech_corpus_building_using_a_modified_greedy_selection) |
| **Festvox `make_nice_prompts`** | Yangi til uchun "nice prompts" topish, keyin eng yaxshi fonetik qoplamaga ega gaplarni ajratish. Festvox/Festival TTS tizimining rasmiy vositasi | [festvox.org/bsv/c2176.html](http://festvox.org/bsv/c2176.html) |
| **Trifon-taqsimoti asosidagi batch greedy** (Alana AI) | Har fonema uchun minimal chastota chegarasi + trifonlarni vokoid/kontoid kategoriyalarga bo'lib taqsimlash. Natija: shu o'lchamdagi namunalarda **55.8% ko'proq turli trifon** (CETUC 12.6%, TTS-Portuguese 12.3% bilan solishtirganda) | [arXiv 2402.05794 "Phonetically rich corpus construction for a low-resourced language"](https://arxiv.org/pdf/2402.05794) |
| **BASPRO** (genetik algoritm) | Greedy'ga alternativa. Almashtirish foizi oshgani sari genetik algoritm greedy'dan ustun kelishi mumkin | [arXiv 2301.04120](https://arxiv.org/pdf/2301.04120) |
| **Bo'g'in qoplamasi greedy** | Kam resursli sharoitda bo'g'in qoplamasini maksimallashtirish → **faqat 1 soat** nutq bilan tushunarli TTS (tasodifiy tanlangan gaplar bilan solishtirganda) | [Tucker & Shalonova, Interspeech 2005](https://www.isca-archive.org/interspeech_2005/tucker05_interspeech.pdf) |

### 4.4 Qancha gap / qancha soat kerak — konkret raqamlar

**Alana AI metodologiyasi (Braziliya portugalchasi, kam resursli til) — eng batafsil ochiq hisob:**

- Maqsad: **20 soat** yozuv
- Gap uzunligi: **10-20 so'z**, o'rtacha **4 bo'g'in** so'zda
- Nutq tezligi: **6 bo'g'in/sekund** (portugalcha uchun o'lchangan)
- Hisob: 20 soat uchun kamida **10,000 gap** kerak (+ zaxira 2,000 gap)
- **Muhim topilma: yangi trifonlar soni ~5,000 gapda to'yinadi** (barqarorlik nuqtasi, nonparametric cumulative sum of squares testi bilan aniqlangan). Undan keyin gaplarni baholash samarasiz
- Natija: shu o'lchamdagi namunalarda 55.8% ko'proq turli trifon

Manba: [arXiv 2402.05794](https://arxiv.org/pdf/2402.05794).

**Gap uzunligi bo'yicha ogohlantirish (shu maqoladan):** uzun gaplar yuqori "richness" bersa ham, yozib olish skripti uchun juda murakkab bo'lib, nutq uzilishlariga (false start, self-correction) olib keladi. Qisqa gaplar esa odatda faqat nominal (sarlavha/mavzu) bo'lib chiqadi va tabiiy nutq uchun yaramaydi. Yechim: gap uchun **minimal va maksimal trifon soni** chegarasini belgilash.

**O'zbek uchun moslashtirish (ishlab chiquvchi o'zi hisoblashi kerak):**

- O'zbek nutq tezligi (bo'g'in/sekund) uchun o'lchangan raqam topilmadi — **o'zbek uchun mavjud emas**. Portugalcha 6 syl/s'ni boshlang'ich taxmin sifatida olish mumkin, lekin FeruzaSpeech'dan o'lchash yaxshiroq
- FeruzaSpeech o'rtacha segment uzunligi **16.39 sekund** (diapazon 3.78–50.69s) — bu TTS uchun **juda uzun**. Aksariyat TTS quvurlari 2-15s segmentlarni kutadi
- Common Voice uz o'rtacha klip uzunligi **4.159 sekund** — TTS uchun **juda qisqa**, prosodiya o'rganish uchun yetarsiz

### 4.5 Amaliy skript tanlash retsepti o'zbek tili uchun

1. **Matn manbasi:** o'zbek Wikipedia (`MLDataScientist/Wikipedia-uzbek-2024-05-01`), yangiliklar (`MLDataScientist/Uzbek_news_dataset`), kitoblar. Domen xilma-xilligi muhim.
2. **Normalizatsiya:** `uztext.normalize()` (apostrof kanonlashtirish) → `uztextnorm` (raqamlar) → o'z qo'shimcha qoidalaringiz (sana/vaqt/qisqartma).
3. **Homoglif tozalash:** Kirill harflarini lotin matnidan ajratib chiqarish (TurkicTTS ogohlantirishi).
4. **G2P:** epitran `uzb-Latn` yoki espeak-ng `uz` → IPA/fonema ketma-ketligi. Ikkisini solishtirib, kelishmagan so'zlarni qo'lda tekshirish.
5. **Trifon hisoblash + greedy tanlash:** har fonema uchun minimal chastota chegarasi, keyin yangi trifon qo'shadigan gaplarni tanlash. **~5,000 gapda to'yinishni kuting.**
6. **Gap filtri:** 10-20 so'z, nutqda o'qishga qulay, akronim va raqamlarsiz (yoki allaqachon kengaytirilgan).
7. **Segment uzunligi maqsadi:** 3-12 sekund (FeruzaSpeech'ning 16.39s o'rtachasidan qisqaroq).

---

## 5. Kam resursli TTS strategiyasi — nashr etilgan dalillar

### 5.1 Qancha soat kerak — konkret soat→MOS/CER dalillari

| Tadqiqot | Til | Ma'lumot miqdori | Baza model | Natija |
|---|---|---|---|---|
| **Amalas et al. 2024/2026** | Darija (Marokash arabchasi) | **1.2 soat** juftlangan (audio+matn) fine-tune + **12 soat** ko'p tilli pretraining (arab+ibroniy+fransuz+golland, har biriga 3 soat) | TransformerTTS → FastSpeech2 (knowledge distillation) + Parallel WaveGAN | **DNSMOS: tushunarlilik 3.72, tabiiylik 3.44** (MONODar2, ko'p tilli). Bir tilli (faqat 12 soat arab) varianti pastroq. Vokoder uchun **6 soat** ishlatilgan | [arXiv 2409.01217](https://arxiv.org/pdf/2409.01217), [Cognitive Computation](https://link.springer.com/article/10.1007/s12559-026-10598-3) |
| **Broadcast + unsupervised data selection** | — | **1 soat** | — | **MOS 4.4 tushunarlilik uchun** (Amalas maqolasida [24] havolasi orqali keltirilgan) |
| **Ko'p tilli VITS** | Yava, Sundan (5 soat); Kxmer (3s); Afrikaans, Nepal, Setsvana, Xosa (2s); Singal (1 soat) | 1-5 soat | VITS | tillar bo'yicha ishlashi ko'rsatilgan |
| **Tucker & Shalonova 2005** | — | **1 soat**, bo'g'in qoplamasi greedy bilan tanlangan | — | tasodifiy tanlashdan ustun, tushunarli TTS | [Interspeech 2005](https://www.isca-archive.org/interspeech_2005/tucker05_interspeech.pdf) |
| **MMS-TTS (Meta)** | 1,100+ til, shu jumladan o'zbek | **~32 soat o'rtacha har til uchun**, **bitta spiker** (odatda erkak), Yangi Ahd o'qishlari | VITS (har til uchun alohida checkpoint) | tarqatilgan modellar | [arXiv 2305.13516](https://arxiv.org/pdf/2305.13516) |

**Muhim eslatma MMS-TTS haqida:** modelning o'qitish ma'lumoti **diniy matn o'qishi** (Yangi Ahd). Bu uslub — sekin, tantanali, deklamatsiya. Ya'ni `mms-tts-uzb-script_cyrillic` odatiy so'zlashuv yoki yangilik uslubidagi nutq bermaydi. Fine-tune qilinganda ham bu prior ta'sir qiladi.

### 5.2 Turkiy qo'shni tillar — eng yaxshi proksi

#### TurkicTTS (Interspeech 2023) — **o'zbek tili aynan baholangan**

Yeshpanov, Mussakhojayeva, Khassanov, "Multilingual Text-to-Speech Synthesis for Turkic Languages Using Transliteration". [arXiv 2305.15749](https://arxiv.org/abs/2305.15749) · [ISCA PDF](https://www.isca-archive.org/interspeech_2023/yeshpanov23_interspeech.pdf) · [GitHub IS2AI/TurkicTTS](https://github.com/IS2AI/TurkicTTS)

**Metod:** Tacotron 2 + WaveGAN, **faqat qozoq tilida** o'qitilgan (KazakhTTS2, Speaker M2, **~58 soat**). Boshqa turkiy tillar uchun **zero-shot**: maqsad tilning harflarini IPA'ga, keyin qozoq alifbosi harflariga xaritalash. 42 IPA belgisi (qozoq alifbosining 42 harfiga mos).

**Natijalar (Table 3) — Q = MOS sifat, C = tushunarlilik (comprehensibility), I = aniqlik (intelligibility, SUS testi):**

| Til | Baholovchi | Q (MOS) | C | I |
|---|---:|---:|---:|---:|
| **Qozoq** (o'qitish tili) | 151 | **4.18** | 97% | 80% |
| Qirg'iz | 14 | 3.54 | 86% | 43% |
| Turk | 18 | 3.25 | 91% | 61% |
| Uyg'ur | 10 | 3.01 | 45% | 26% |
| Ozarbayjon | 47 | 2.93 | 90% | 52% |
| Saxa (Yoqut) | 254 | 2.85 | 93% | 15% |
| **O'ZBEK** | **22** | **2.85** | **80%** | **45%** |
| Tatar | 15 | 2.82 | 79% | 17% |
| Boshqird | 11 | 2.67 | 92% | 47% |
| Turkman | 6 | 2.37 | 67% | 57% |
| **Jami** | 548 | **3.25** | 92% | 41% |

**Ishlab chiquvchi uchun xulosa:** zero-shot qozoq→o'zbek **MOS 2.85** beradi, ya'ni "o'rtachadan past". O'z tilida o'qitilgan qozoq esa **4.18**. **Farq 1.33 MOS** — bu haqiqiy o'zbek ma'lumotida fine-tune qilishning qiymatini ko'rsatadi. Maqola o'zi aytadi: "we believe that performance can be further improved for all target languages by **fine-tuning the pre-trained models with the seed data** of the corresponding language."

#### KazakhTTS / KazakhTTS2 — turkiy til uchun oltin standart

- **KazakhTTS**: 93 soat, 2 spiker ([arXiv 2104.08459](https://arxiv.org/pdf/2104.08459))
- **KazakhTTS2**: **271 soat**, **5 spiker** (3 ayol, 2 erkak), kitob + Wikipedia bilan mavzu qoplamasi kengaytirilgan. **MOS 3.6–4.2** barcha 5 ovoz uchun. Tacotron 2 (encoder: 1× biLSTM 512 unit; decoder: 2× uniLSTM 1024 unit). ([arXiv 2201.05771](https://arxiv.org/pdf/2201.05771), [GitHub IS2AI/Kazakh_TTS](https://github.com/IS2AI/Kazakh_TTS), [LREC 2022](https://aclanthology.org/2022.lrec-1.578.pdf))
- Piper'da qozoq ovozi bor (`kk_KZ`: issai — high quality), o'zbekda yo'q

#### Turkiy tillar uchun boshqa ishlar

- **Turkman**: ["From Speech to Summary in Turkmen: A Parameter-Efficient Neural Pipeline"](https://www.mdpi.com/2076-3417/16/12/5734) — MMS language adapter'ni **turk tilidan** initsializatsiya qilish + LoRA PEFT keskin ma'lumot yetishmovchiligida raqobatbardosh natija beradi. **Bu o'zbek uchun ham to'g'ridan-to'g'ri qo'llanadigan strategiya** (turk → o'zbek).
- **Turk**: Piper'da 3 ovoz (`tr_TR`: dfki, fahrettin, fettah — medium), XTTS-v2'da qo'llanadi, `facebook/mms-tts-tur` bor. Turk tili turkiy tillar orasida eng ko'p resursli — transfer learning uchun manba til sifatida eng yaxshi tanlov.
- **Ozarbayjon, Qirg'iz**: Piper ovozi yo'q. `facebook/mms-tts-azj-script_latin` bor.
- **Umumiy muammo** (TurkicTTS maqolasidan): "most of the existing datasets are **proprietary**, while the majority of available datasets are either of **low quality** or **unsuitable** for building state-of-the-art TTS architectures."

### 5.3 O'zbek uchun tavsiya etiladigan strategiya (dalillarga asoslangan)

Dalillarga ko'ra, kamayib boruvchi tartibda:

1. **FeruzaSpeech'dan boshlash** (60 soat, bitta ayol spiker, Lotin+Kirill, CC BY 4.0) — bu allaqachon TTS-uslubidagi o'qib berilgan nutq. Segmentlarni 16.39s o'rtachadan qisqartirish kerak.
2. **Baza model uchun ikki variant:**
   - `facebook/mms-tts-uzb-script_cyrillic` — o'zbek priori bor, lekin **Kirill** + **CC-BY-NC** (tijoriy foydalanish mumkin emas!) + diniy o'qish uslubi
   - `facebook/mms-tts-tur` yoki turk VITS/Piper ovozi — Lotin yozuvi, ko'proq ma'lumot, turkiy fonologiya yaqin, litsenziya yaxshiroq. Turkman ishi bu yo'lni tasdiqlaydi
3. **Minimal maqsad: 1-5 soat** yuqori sifatli bitta spiker ma'lumoti (Darija 1.2 soat bilan DNSMOS 3.72 oldi). **Yaxshi maqsad: 10-20 soat** greedy-tanlangan trifon-boy skript bilan.
4. **CER back-check:** `Kotib/uzbek_stt_v1` (CER 7.0%) yoki `GigaAM Multilingual Large` (Feruza'da 5.93% WER). **ASR shovqinini hisobga oling** — o'qib berilgan toza nutqda ham eng yaxshi model ~5% WER beradi, shuning uchun CER chegarasini juda qattiq qo'ymaslik kerak.

---

## 6. "O'zbek uchun mavjud emas" — aniq ro'yxat

Bu bo'lim eng qimmatli, chunki ishlab chiquvchi vaqtini bekorga sarflamasligi kerak.

| Nima | Holat | Manba |
|---|---|---|
| **MMS-TTS Lotin o'zbek** (`mms-tts-uzb-script_latin`, `mms-tts-uzn`) | ❌ **MAVJUD EMAS** — faqat Kirill | [HF qidiruv](https://huggingface.co/facebook/mms-tts-uzb-script_cyrillic) |
| **Piper o'zbek ovozi** (`uz_UZ`) | ❌ **MAVJUD EMAS** — VOICES.md'da yo'q | [piper VOICES.md](https://github.com/rhasspy/piper/blob/master/VOICES.md) |
| **XTTS-v2 o'zbek** | ❌ **QO'LLANMAYDI** — 17 tilda yo'q (turk bor) | [HF coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) |
| **NeMo-text-processing o'zbek TN/ITN** | ❌ **MAVJUD EMAS** | [GitHub NVIDIA/NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing) |
| **VoxPopuli o'zbek** | ❌ **MAVJUD EMAS** — 23 EI tili | [arXiv 2101.00390](https://ar5iv.labs.arxiv.org/html/2101.00390) |
| **CharsiuG2P o'zbek** | ❌ **BOR, LEKIN NOTO'G'RI** — mualliflar "do not use it" deydi | [GitHub CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P) |
| **Whisper zero-shot o'zbek** | ❌ **AMALDA FOYDASIZ** — WER 90.2% | [Whisper maqolasi](https://arxiv.org/pdf/2212.04356), [BlueRaccoon karta](https://huggingface.co/BlueRaccoon/whisper-small-uz) |
| **O'zbek talaffuz lug'ati (CMUdict uslubida)** | ❌ **ochiq mavjud emas** | qidiruvda topilmadi |
| **Sana/vaqt/telefon normalizatsiya kutubxonasi** | ❌ **MAVJUD EMAS** — `uztextnorm` faqat raqam/valyuta/foiz/kasr/rim | [PyPI uztextnorm](https://pypi.org/project/uztextnorm/) |
| **O'zbek qisqartma/akronim kengaytirish resursi** | ❌ **MAVJUD EMAS** | qidiruvda topilmadi |
| **O'zbek nutq tezligi (bo'g'in/sekund) o'lchovi** | ❌ **nashr etilmagan** | qidiruvda topilmadi |
| **O'zbek ko'p spikerli ochiq TTS korpusi** | ❌ **MAVJUD EMAS** — faqat FeruzaSpeech (1 spiker) | — |
| **O'zbek TTS uchun MOS benchmark / leaderboard** | ❌ **MAVJUD EMAS** | — |
| **espeak-ng o'zbek aniqlik o'lchovi** | ❌ **o'lchanmagan** — `status testing` | [voice fayli](https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/espeak-ng-data/lang/trk/uz) |
| **epitran o'zbek aniqlik o'lchovi** | ❌ **o'lchanmagan** | — |
| **2026 alifbosida matn korpusi** | ❌ **deyarli yo'q** — UzTransliterator mualliflari "there is no publicly available data for the New Latin alphabet yet" deydi (2022) | [CEUR Vol-3315](https://ceur-ws.org/Vol-3315/paper05.pdf) |

---

## 7. Qarama-qarshi manbalar

| Masala | Manba A | Manba B | Izoh |
|---|---|---|---|
| **USC litsenziyasi** | CC BY 4.0 — [ISSAI e'loni](https://issai.nu.edu.kz/2021/08/06/issai-team-has-developed-the-uzbek-speech-corpus/) va arXiv sahifasi | MIT — [HF dataset kartasi](https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus) | Tijoriy ishlatishdan oldin ISSAI bilan yozishma kerak |
| **USC hajmi** | 105 soat / 958 spiker — [arXiv 2107.14419](https://arxiv.org/abs/2107.14419) | "~900 soat" da'vosi | 958 = spiker soni, soat emas. ~900 soat aslida UzbekVoice |
| **NVIDIA FastConformer uz WER (CV)** | 16.46% (punktuatsiya bilan) / 7.78% (punktuatsiyasiz) — [model kartasi](https://huggingface.co/nvidia/stt_uz_fastconformer_hybrid_large_pc) | 8.86% — [alphacep benchmark](https://github.com/alphacep/awesome-speech/blob/main/uzbek.md) | Turli normalizatsiya/dekodlash sozlamalari; benchmark odatda punktuatsiyasiz o'lchaydi |
| **O'zbek unlilar ro'yxati** | /i, e, ɜ, a, o, u/ — Ido 2025 JIPA | /a, æ, i, o, u, ø/ — boshqa manbalar | **Soni bir xil: 6** |
| **O'zbek undoshlar soni** | ~22 (Ido 2025'dan olingan ro'yxat, to'liq bo'lmasligi mumkin) | 25 (Shimoliy o'zbek, Wikipedia/ResearchGate) | ~24-25 deb olish xavfsiz |
| **DavronSherbaev/uzbekvoice yozuvi** | HF kartasidagi misollar **Lotin** (`Agar biz chiqarayotgan qaror noto'g'ri bo'lsa...`) | `ai4uz/uzbekvoice-filtered` uchun avtomatik tahlil "Cyrillic" dedi | **Ishlab chiquvchi o'zi tekshirishi kerak** — datasetni yuklab olib, birinchi 100 qatorni ko'rish |

---

## 8. Manbalar ro'yxati

**Datasetlar va korpuslar:**
- USC: https://arxiv.org/abs/2107.14419 · https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus · https://github.com/IS2AI/Uzbek_ASR · https://issai.nu.edu.kz/uzbek-asr/
- Common Voice uz: https://mozilladatacollective.com/datasets/cmn29fa4y0150o1073fqww73p · https://github.com/common-voice/cv-dataset/
- FeruzaSpeech: https://arxiv.org/abs/2410.00035 · https://arxiv.org/html/2410.00035v1 · https://aclanthology.org/2024.icnlsp-1.37.pdf · https://huggingface.co/datasets/k2speech/FeruzaSpeech
- UzbekVoice: https://huggingface.co/datasets/DavronSherbaev/uzbekvoice · https://huggingface.co/datasets/ai4uz/uzbekvoice-filtered · https://corpus.uzbekvoice.ai/en-US/resus · https://discourse.mozilla.org/t/creating-open-and-accessible-resources-for-uzbek-the-uzbekvoice-ai-project/112634
- FLEURS: https://huggingface.co/datasets/google/fleurs · https://ar5iv.labs.arxiv.org/html/2205.12446
- MMS: https://arxiv.org/pdf/2305.13516 · https://jmlr.org/papers/volume25/23-1318/23-1318.pdf
- CMU Wilderness: http://festvox.org/cmu_wilderness/ · https://github.com/festvox/datasets-CMU_Wilderness
- VoxPopuli: https://ar5iv.labs.arxiv.org/html/2101.00390
- O'zbek resurs ro'yxati + WER benchmark: https://github.com/alphacep/awesome-speech/blob/main/uzbek.md
- islomov: https://huggingface.co/datasets/islomov/news_youtube_uzbek_speech_dataset · .../it_youtube_uzbek_speech_dataset · .../podcasts_tashkent_dialect_youtube_uzbek_speech_dataset
- Abduqayum 780h: https://huggingface.co/datasets/Abduqayum/Uzbek-STT-Dataset-780h
- Biruniy (tijoriy): https://www.biruniy.uz/

**Modellar:**
- MMS-TTS uzb: https://huggingface.co/facebook/mms-tts-uzb-script_cyrillic
- TurkicTTS: https://github.com/IS2AI/TurkicTTS · https://arxiv.org/abs/2305.15749 · https://www.isca-archive.org/interspeech_2023/yeshpanov23_interspeech.pdf
- Piper: https://github.com/rhasspy/piper/blob/master/VOICES.md · https://huggingface.co/rhasspy/piper-voices
- XTTS-v2: https://huggingface.co/coqui/XTTS-v2
- Kotib STT: https://huggingface.co/Kotib/uzbek_stt_v1
- NVIDIA FastConformer uz: https://huggingface.co/nvidia/stt_uz_fastconformer_hybrid_large_pc
- Sayro TTS: https://huggingface.co/uzlm/sayro-tts-1.7B
- Whisper: https://arxiv.org/pdf/2212.04356 · https://huggingface.co/BlueRaccoon/whisper-small-uz
- KazakhTTS2: https://arxiv.org/pdf/2201.05771 · https://github.com/IS2AI/Kazakh_TTS · https://aclanthology.org/2022.lrec-1.578.pdf

**Yozuv va matn:**
- Alifbo: https://en.wikipedia.org/wiki/Uzbek_alphabet · https://www.gazeta.uz/en/2026/09/10/uzb-alphabet/ · https://www.gazeta.uz/en/2026/07/09/alphabet/ · https://timesca.com/uzbekistan-latin-alphabet-changes-senate/
- Apostrof: https://en.wikipedia.org/wiki/O%CA%BB · https://en.wikipedia.org/wiki/G%CA%BB
- Transliteratsiya: https://ceur-ws.org/Vol-3315/paper05.pdf · https://arxiv.org/abs/2205.09578 · https://github.com/UlugbekSalaev/UzTransliterator · https://github.com/abafaboy/uztext
- Normalizatsiya: https://pypi.org/project/uztextnorm/ · https://gitlab.com/adizbek/uztextnorm · https://github.com/NVIDIA/NeMo-text-processing
- Turkiy script unification: https://arxiv.org/pdf/2608.09356

**Fonologiya va G2P:**
- Ido 2025 JIPA: https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/uzbek/48B3BBA8C522A61CEAE0E21193B3C779
- espeak-ng: https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md · https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/dictsource/uz_rules · https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/espeak-ng-data/lang/trk/uz
- epitran: https://github.com/dmort27/epitran · https://aclanthology.org/L18-1429.pdf
- CharsiuG2P: https://github.com/lingjzhu/CharsiuG2P

**Skript tanlash va kam resursli TTS:**
- Alana AI: https://arxiv.org/pdf/2402.05794
- BASPRO: https://arxiv.org/pdf/2301.04120
- Festvox: http://festvox.org/bsv/c2176.html
- Tucker & Shalonova: https://www.isca-archive.org/interspeech_2005/tucker05_interspeech.pdf
- Amalas et al.: https://arxiv.org/pdf/2409.01217 · https://link.springer.com/article/10.1007/s12559-026-10598-3
- Turkman PEFT: https://www.mdpi.com/2076-3417/16/12/5734

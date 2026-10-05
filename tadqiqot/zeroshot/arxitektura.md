# Zero-shot ovoz klonlash: MEXANIZM

> **Bu hujjatning aniq mavzusi:** "10 sekund ovoz berasan — o'sha ovozda gapiradi" degan narsa **ichidan qanday ishlaydi**.
> Bu yerda model ro'yxati, VRAM hisobi, o'zbek tili masalasi yoki arzon o'qitish usullari **yo'q** — ular boshqa hujjatlarda.
> Bu yerda faqat **mashina qanday qurilgan**.

Oxirgi yangilanish: 2026-10-01

---

## 0. Eng qisqa javob (TL;DR)

Zero-shot ovoz klonlash — bu **"ovozni o'rganish" emas**. Hech qanday o'qitish bo'lmaydi. Model allaqachon
minglab odamning ovozida gapirishni o'rgangan bo'ladi; sen bergan 10 sekund esa shu modelga **qaysi ovozni
tanlashini ko'rsatadigan ishora** bo'ladi.

Shu ishorani modelga uzatishning tarixda **ikkita tubdan boshqa yo'li** bor:

| | **(a) Speaker embedding yo'li** | **(b) In-context / audio-prompt davomi** |
|---|---|---|
| G'oya | Ovozni **bitta vektorga siqib** modelga beramiz | Referens audioni **boshlanishi sifatida** beramiz, model shuni **davom ettiradi** |
| Boshlanishi | SV2TTS, Google, 2018 ([arXiv:1806.04558](https://arxiv.org/abs/1806.04558)) | VALL-E, Microsoft, 2023 ([arXiv:2301.02111](https://arxiv.org/abs/2301.02111)) |
| Referens → modelga yetadigan ma'lumot | ~256 son (bir vektor) | 3 sekund uchun ~1800 token (VALL-E'da) |
| Alohida "speaker" tushunchasi bormi | Bor — maxsus tarmoq | **Yo'q.** Modelda "speaker" degan narsa umuman yo'q |
| Natija | "O'xshash, lekin aynan o'sha emas" | Ancha yaqin, lekin beqarorroq |

Eng muhim raqam — **bir xil arxitektura, bir xil ma'lumotda** faqat shu ikki yo'lni almashtirib ko'rgan
nazoratli tajriba (P-Flow, NeurIPS 2023):

```
P-Flow (W/O PROMPT)  = qat'iy o'lchamli speaker embedding   →  SECS 0.373
P-Flow (PROMPT)      = audio-prompt in-context              →  SECS 0.544
Ground truth                                                →  SECS 0.64
```

Ya'ni boshqa hech narsani o'zgartirmasdan, faqat "vektor" o'rniga "prompt" qo'yilganda ovoz o'xshashligi
**0.373 → 0.544** bo'ldi. Bu — butun soha 2023-yilda (b) yo'liga o'tib ketganining sababi.
Manba: [P-Flow, NeurIPS 2023, Table 1](https://papers.nips.cc/paper_files/paper/2023/file/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Paper-Conference.pdf)

---

# 1-BOB. Model 10 sekundni qanday "iste'mol" qiladi — ikki yo'l

## 1.1. Yo'l (a): Speaker encoder / speaker embedding

### Diagramma

```
   ╔══════════════════════════════════════════════════════════════════════╗
   ║  YO'L (a):  OVOZNI BITTA VEKTORGA SIQISH                             ║
   ╚══════════════════════════════════════════════════════════════════════╝

  10 sek referens audio
  ┌──────────────────┐
  │ ~~~~~~~~~~~~~~~~ │   (faqat ovoz, matn kerak emas)
  └────────┬─────────┘
           │  40-kanalli log-mel
           v
   ┌───────────────────────────┐
   │   SPEAKER ENCODER         │  ← ALOHIDA tarmoq, ALOHIDA o'qitilgan
   │   3 x LSTM (768 hujayra)  │     Vazifasi: TTS emas, SPEAKER VERIFICATION
   │   → 256-o'lchovli proj.   │     Loss: GE2E (arXiv:1710.10467)
   │   800ms oyna, 50% ustma-  │     O'qitish: 36M gap, 18K spiker
   │   ust, o'rtacha + L2-norm │
   └───────────┬───────────────┘
               │
               v
      ┌─────────────────┐
      │  d-vector       │   ← MANA BU — BUTUN "BOTTLENECK".
      │  [256 float]    │      10 sekund ovozdan qolgan hammasi — shu 256 son.
      └────────┬────────┘
               │
   matn ───────┼──────────────────────────────────┐
     │         │                                  │
     v         │                                  │
 ┌───────────────────────┐                         │
 │ Synthesizer encoder   │                         │
 │ chiqishi:             │                         │
 │  h1 h2 h3 ... hT      │                         │
 └───────────────────────┘                         │
      │  │  │       │                              │
      v  v  v       v    ← HAR BIR time-step'ga    │
   [h1|e][h2|e][h3|e]... ← embedding CONCAT        │
      │                                            │
      v                                            │
 ┌─────────────────────┐      ┌──────────────┐     │
 │ Attention + decoder │ ───> │  mel-spek.   │     │
 └─────────────────────┘      └──────┬───────┘     │
                                     v             │
                              ┌─────────────┐      │
                              │  VOCODER    │ <────┘ (ba'zi modellarda
                              │  (WaveNet / │          vocoder ham embedding
                              │   HiFi-GAN) │          oladi)
                              └──────┬──────┘
                                     v
                                  tovush
```

### Mexanizm: nega bu ishlaydi?

Kalit g'oya — **transfer learning**, ya'ni boshqa vazifa uchun o'qitilgan tarmoqni qarzga olish.
SV2TTS (Jia va boshqalar, 2018) quyidagini qildi:

1. Speaker encoder **TTS uchun emas**, **speaker verification** (ovozni tanish) uchun o'qitiladi.
   Uning loss funksiyasi — **GE2E** (Generalized End-to-End, [arXiv:1710.10467](https://arxiv.org/abs/1710.10467)).
2. GE2E bir batch ichidagi har bir gapning embedding'ini **barcha spikerlarning markazlariga** nisbatan
   o'lchaydi va o'zining markaziga tortadi, boshqalardan uzoqlashtiradi. GE2E maqolasi TE2E'ga qaraganda
   "EER'ni 10%'dan ko'proq kamaytiradi va o'qitish vaqtini 60%'ga qisqartiradi" deydi.
3. Natijada hosil bo'lgan 256-o'lchovli fazo — **"ovozlar fazosi"**. Unda bitta nuqta = bitta tembr.
4. Keyin bu fazo **muzlatiladi** va synthesizer unga shart (conditioning) sifatida o'qitiladi.

**Nega ko'rilmagan (unseen) spikerga ham ishlaydi?** Chunki speaker encoder 18 000 spikerda o'qitilgan
va u yodlagan emas, balki **uzluksiz fazo** qurgan. Yangi odam ham shu fazoning biror nuqtasiga tushadi —
qo'shni nuqtalarni synthesizer allaqachon ko'rgan, shuning uchun interpolatsiya qilib beradi.
SV2TTS buni shunday ifodalaydi: *"training on a speaker discrimination task leads to an embedding which is
directly suitable for conditioning the synthesis network on speaker identity"*, va umumlashtirish qobiliyati
*"is based on the quality of the representation learned by the speaker encoder"*.

**Buni isbotlaydigan eng aniq jadval** (SV2TTS, Table 5) — speaker encoder'ni necha spikerda o'qitganiga
qarab sintez qilingan ovozning speaker-verification EER'i:

| Encoder o'qitilgan spiker soni | Sintez ovozning SV-EER (unseen) |
|---|---|
| 1 200 | 16.60 % |
| 8 400 | 10.14 % |
| 18 000 | **5.08 %** |

Bu — "ko'p spiker kerak" degan gapning eng toza, eng eski raqamli dalili.

### Embedding qanday kiritiladi (injection)?

Modellar bo'yicha aniq joylar:

- **SV2TTS (2018):** *"An embedding vector for the target speaker is concatenated with the synthesizer
  encoder output at each time step."* Asosiy modelda embedding **256-o'lchovli** (kichik datasetlarda 64).
  ([arXiv:1806.04558](https://arxiv.org/abs/1806.04558))
- **YourTTS (2022):** H/ASP encoder embedding'i VITS'ning bir nechta modulida ishlatiladi —
  coupling layer'lar va posterior encoder'da *global conditioning*, duration predictor'da
  matn-encoder chiqishiga **qo'shiladi** (linear projection bilan o'lcham tenglashtirilgach),
  vocoder'dan oldin decoder chiqishiga qo'shiladi. ([arXiv:2112.02418](https://arxiv.org/abs/2112.02418))
- **XTTS (2024):** ikki yo'ldan: (1) Conditioning Encoder referens mel'ni **32 ta 1024-o'lchovli
  embedding**ga aylantiradi (6 qatlamli 16-boshli attention + Perceiver Resampler) va GPT-2'ni shartlaydi;
  (2) HiFi-GAN decoder'ga H/ASP embedding **har bir upsampling qatlamida linear projection orqali
  qo'shiladi**. ([arXiv:2406.04904](https://arxiv.org/abs/2406.04904))
  — Diqqat: XTTS bitta vektordan **32 vektorga** o'tgan. Bu allaqachon (a) va (b) o'rtasidagi ko'prik.
- **NaturalSpeech 3 / FACodec (2024):** tembr **alohida global vektor** sifatida ajratiladi
  (Transformer encoder speech-encoder chiqishini bitta global vektorga aylantiradi) va decoder'da
  **conditional layer normalization** orqali qo'shiladi. Zero-shot voice conversion'da shunchaki
  *"extract the speaker embedding from the prompt speech to replace the speaker embedding from the
  source speech"*. ([arXiv:2403.03100](https://arxiv.org/abs/2403.03100))
- **OpenVoice (2023):** bu eng qiziq variant — **ayirish-qo'shish** sxemasi.
  Normalizing flow tembrni **olib tashlaydi** (flow phoneme-darajali matn reprezentatsiyasiga
  moslashtirib o'qitiladi, shuning uchun chiqishda tembr qolmaydi), keyin **teskari flow** maqsadli
  spikerning tembr embedding'i bilan shartlanib tembrni **qaytarib qo'yadi**.
  HiFi-GAN to'lqinga aylantiradi. ([arXiv:2312.01479](https://arxiv.org/abs/2312.01479))

### Speaker encoder'larning evolyutsiyasi

| Nom | Yil | Qisqacha | Manba |
|---|---|---|---|
| d-vector (LSTM + GE2E) | 2017–18 | 3×LSTM(768) → 256-dim, L2-norm | [arXiv:1710.10467](https://arxiv.org/abs/1710.10467) |
| x-vector (TDNN) | 2018 | TDNN + statistics pooling | SV2TTS'da muqobil sifatida tilga olingan |
| **ECAPA-TDNN** | 2020 | 1D Res2Net + SE-block + **channel-dependent attentive statistics pooling** + multi-layer feature aggregation; **192-dim** embedding; VoxCeleb1 **EER 0.87 %**, minDCF 0.1066 (C=1024) | [arXiv:2005.07143](https://arxiv.org/abs/2005.07143) |
| WavLM-TDNN | — | Bugun ko'pchilik **SIM-o / SECS metrikasi** shu bilan hisoblanadi | VALL-E, P-Flow'da ishlatilgan |

> Diqqat: **bugun ECAPA/WavLM asosan o'lchov asbobi** (SIM-o metrikasi) sifatida ishlatiladi,
> generatsiya mexanizmi sifatida emas. Ya'ni (a) yo'li metrikaga aylandi.

### Yo'l (a)ning HUJJATLASHTIRILGAN CHEGARALARI — "nega o'xshash, lekin aynan o'sha emas?"

Bu savolga javob texnik jihatdan juda aniq. Uchta sabab bor.

**1-sabab: Informatsion bottleneck.**
SV2TTS o'zi yozadi: model *"is constrained by the small capacity of the speaker embedding"* —
va shuning uchun **5 sekunddan uzun referens berish yordam bermaydi**. 10 sekund bersang ham, 10 daqiqa
bersang ham modelga baribir 256 son yetadi. Qolgan hammasi yo'qoladi.

Solishtirish uchun hisob:
```
Yo'l (a):  256 float32                             ≈   1 KB,  lekin ma'lumot sifatida
                                                       bundan ham kam (L2-norm + diskriminativ)
Yo'l (b):  3 sek × 75 Hz × 8 kodkitob × 10 bit     ≈ 2.25 KB TOZA diskret ma'lumot
           = 1800 token, har biri 1024 variantdan
```
Bu — bir vektor emas, **to'liq audio yozuvi**. Mana shuning uchun (b) aniqroq.

**2-sabab: GE2E loss prosodiyani ATAYLAB o'chiradi.**
Bu eng muhim, lekin kam tushuntiriladigan nuqta. GE2E bir spikerning **barcha** gaplarini bitta
markazga tortadi. Lekin bir odamning gaplari prosodiya, emotsiya, tezlik, mikrofon bo'yicha bir-biridan
farq qiladi. Demak GE2E loss modelga aytadi: **"bu farqlarni tashla, ular shovqin"**.
Natijada embedding **prosodiyaga invariant** bo'lib o'qitiladi — ya'ni prosodiya ma'lumoti
bottleneck'dan o'tolmaydi. Dizayn bo'yicha shunday.

SV2TTS buni o'z og'zi bilan tasdiqlaydi: unseen spikerlarda similarity pasayishini
*"suggesting that some nuances, e.g. related to characteristic prosody, are lost"* deb izohlaydi.

**3-sabab: lekin baribir biroz oqib o'tadi (leakage).**
Qizig'i shu — SV2TTS bir vaqtda teskarisini ham aytadi:
*"The model is also not able to completely isolate the speaker voice from the prosody of the reference
audio"* — ya'ni sintez ba'zan referensning prosodiyasini **taqlid qiladi**. Demak ajratish ideal emas:
kerakli narsa (tembrning nozik qirralari) o'tmaydi, kerak bo'lmagan narsa (referens prosodiyasi) oqadi.

**Raqamlar — bo'shliqni ko'rsatadi** (SV2TTS, Similarity MOS):

| Dataset | **Seen** spiker | **Unseen** spiker | Bo'shliq |
|---|---|---|---|
| VCTK | 4.22 ± 0.06 | **3.28 ± 0.07** | −0.94 |
| LibriSpeech | 3.28 ± 0.08 | **3.03 ± 0.09** | −0.25 |

Naturalness esa unseen'da ham yaxshi (VCTK 4.20, LibriSpeech 4.12) — ya'ni **"tabiiy gapiradi,
lekin boshqa odam ovozida"**. Bu (a) yo'lining klassik imzosi.

**4-sabab: aksent.** SV2TTS'da speaker encoder faqat Shimoliy Amerika aksentida o'qitilgan edi —
britancha VCTK spikerlarida sifat pasaydi. Ya'ni embedding fazosi **o'zi ko'rmagan aksent turini
yomon kodlaydi**. Bu o'zbek tili uchun bevosita ogohlantirish.

---

## 1.2. Yo'l (b): In-context learning / audio-prompt davomi

### Diagramma

```
   ╔══════════════════════════════════════════════════════════════════════╗
   ║  YO'L (b):  REFERENSNI "BOSHLANISH" QILIB BERIB, DAVOMINI SO'RASH    ║
   ╚══════════════════════════════════════════════════════════════════════╝

   3 sek referens audio              referens matni        yangi matn
   ┌──────────────────┐             "hello there"       "o'zbek tilida..."
   │ ~~~~~~~~~~~~~~~~ │                   │                   │
   └────────┬─────────┘                   │                   │
            │ NEURAL CODEC                └─────────┬─────────┘
            │ (EnCodec)                             │
            v                                       v
   ┌──────────────────────┐              ┌────────────────────────┐
   │ diskret tokenlar     │              │ matn tokenlari (butun) │
   │ C̃ = [312, 88, 901,  │              │ x = [h,e,l,l,o,...]    │
   │      45, ... ]       │              └───────────┬────────────┘
   └──────────┬───────────┘                          │
              │                                      │
              └──────────────┬───────────────────────┘
                             v
      ╔══════════════════════════════════════════════════════════╗
      ║        BITTA TRANSFORMER (GPT kabi)                      ║
      ║                                                          ║
      ║  INPUT:  [matn tokenlari] [referens audio tokenlari] ──> ?║
      ║                                ^^^^^^^^^^^^^^^^           ║
      ║                                PREFIX / PROMPT            ║
      ║                                                          ║
      ║  VAZIFA: keyingi audio tokenni ayt. Keyingisini. Keyin... ║
      ╚══════════════════════════════════════════════════════════╝
                             │
                             v
   ┌───────────────────────────────────────────────────────────┐
   │ C̃ (berilgan)        │  C (yangi, model yozdi)            │
   │ 312, 88, 901, 45... │  77, 420, 12, 668, 203, ...        │
   └─────────────────────┴───────────────────────────────────┬─┘
                                                             │
                                          faqat yangi qism ──┘
                                                 v
                                    ┌─────────────────────┐
                                    │  CODEC DECODER      │
                                    └──────────┬──────────┘
                                               v
                                            tovush


   ★ MODEL ICHIDA "SPEAKER" DEGAN NARSA YO'Q.
   ★ SPEAKER ENCODER YO'Q. EMBEDDING YO'Q. KLONLASH MODULI YO'Q.
```

### KALIT TUSHUNCHA: nega davom ettirish ovozni beradi?

Bu eng muhim konseptual qadam. Sekin boramiz.

**1-qadam: model nimaga o'qitilgan?**
VALL-E o'qitish davrida **faqat bitta** vazifani ko'radi:

> *"Matnni va shu paytgacha bo'lgan audio tokenlarni bilib, keyingi audio tokenni top."*

Boshqa hech narsa. "Spiker" so'zi loss funksiyasida yo'q. Hech kim modelga "mana bu Ahmad, mana bu
Dilnoza" demaydi.

**2-qadam: audio tokenlarda nima bor?**
Audio tokenlar — bu tovushning siqilgan yozuvi. Ularda **hamma narsa** bor:
kim gapiradi, qanday tembr, qanday balandlik, qanday tezlik, hatto xonaning akustikasi va shovqini.
Codec hech narsani ajratmaydi — u shunchaki tovushni siqadi.

**3-qadam: optimal strategiya nima?**
Mana bu yerda sehr bo'ladi. Model 1000-tokenni topishi kerak, 1..999-tokenlarni ko'rib turadi.
**Haqiqiy yozuvlarda gap o'rtasida odam o'zgarmaydi.** Hech qachon. Shuning uchun loss'ni
minimallashtiradigan yagona to'g'ri strategiya:

> **"1..999-tokenlar bilan akustik jihatdan BIR XIL bo'lgan tokenni ayt."**

Model **buni o'rganishga majbur** — boshqa yo'l bilan loss pasaymaydi. Ya'ni model
**"ovozni ushlab turish" qoidasini ma'lumotdan o'zi chiqarib oladi**. Bu dizayner qo'ygan
mexanizm emas, bu **ma'lumotning statistik xususiyatidan kelib chiqqan majburiy xulq**.

**4-qadam: inference'da aldaymiz.**
Endi biz modelga 1..999-token o'rniga **butunlay boshqa odamning** 3 sekundlik yozuvini beramiz.
Model o'zining o'rgangan qoidasiga rioya qiladi — "oldingi tokenlar bilan bir xil bo'l" —
va o'sha notanish odamning ovozida davom etadi.

> **Klonlash — maxsus funksiya emas. Bu modelning "izchillik" qoidasining YON TA'SIRI.**
> Model nima qilayotganini bilmaydi. U shunchaki o'z ishini davom ettiradi.

VALL-E'da bu rasmiy ravishda shunday yoziladi:

```
p(c_{:,1} | x, C̃_{:,1}; θ_AR) = ∏_t  p( c_{t,1} | c_{<t,1},  C̃_{:,1},  x )
                                            ^^^^^^^  ^^^^^^^   ^
                                            o'zi      PREFIX   matn
                                            yozgani  (referens)
```
va maqola aniq aytadi: *"the acoustic token sequence of the enrolled recording is used as the
prefix in AR decoding"*, hamda *"only c_{:,1} is predicted while the prefix c̃_{:,1} is given during
inference"*. ([arXiv:2301.02111](https://arxiv.org/abs/2301.02111))

### Analogiya (bu eng tez tushuntiradi)

**Yo'l (a) = politsiya suratchisi.**
Guvoh odamning yuzini 256 so'z bilan tasvirlaydi ("burni to'g'ri, ko'zi qo'ng'ir..."), suratchi shu
tasvir bo'yicha rasm chizadi. Natija — **tanib olinadi, lekin aynan o'sha odam emas**. Chunki
tasvirlash nuqtasida ma'lumot yo'qolgan. Va tasvir qoidalari (GE2E) ataylab "o'tkinchi" narsalarni
(kayfiyat, yorug'lik) tashlab yuborishni o'rgatgan.

**Yo'l (b) = ksero.**
Original varaq shishaning ustida **jismonan yotadi**. Mashina uni ko'rib turib davomini chizadi.
Hech kim hech narsani tasvirlab bermadi.

**Yoki GPT bilan:**
ChatGPT'ga Navoiy g'azalining birinchi baytini berib "davom et" desang, u o'sha uslubda davom etadi.
Hech kim "Navoiy uslubi vektori" o'qitgan emas. U shunchaki **prefix'ga mos keladi**.
Yo'l (b) — xuddi shu narsa, faqat tokenlar **tovush** bo'lgan holda.

### (a) va (b)ni nazoratli taqqoslash

#### Eng toza dalil: P-Flow ablatsiyasi (NeurIPS 2023)

Bir xil arxitektura, bir xil ma'lumot (LibriTTS 260 soat), bir xil decoder, bir xil o'qitish.
**Faqat shartlash (conditioning) usuli almashtirilgan:**

| Variant | WER ↓ | **SECS ↑** |
|---|---|---|
| Ground truth (HiFi-GAN qayta-sintez) | 2.4 | 0.64 |
| **P-Flow (w/o prompt)** = qat'iy o'lchamli speaker embedding | 2.9 | **0.373** |
| **P-Flow** = audio prompt (in-context) | 2.6 | **0.544** |

Maqolaning xulosasi: *"the use of speech prompts leads to a substantial enhancement in speaker
similarity"* va prompting usuli *"is more effective for zero-shot TTS compared to encoding all the
speaker information into a fixed-size vector"*.

#### Sohada kuzatilgan raqamlar

| Model | Yo'l | O'qitish ma'lumoti | WER ↓ | SIM / SECS ↑ | Manba |
|---|---|---|---|---|---|
| SV2TTS (2018) | (a) | 436 soat / 1172 spiker (synth) | — | SMOS 3.03–3.28 (unseen) | [1806.04558](https://arxiv.org/abs/1806.04558) |
| YourTTS (2022) | (a) | 500+ soat | 7.7 | 0.337 | [2112.02418](https://arxiv.org/abs/2112.02418) |
| VALL-E (2023) | (b) | 60 000 soat / 7000+ spiker | 5.9 | 0.580 | [2301.02111](https://arxiv.org/abs/2301.02111) |
| VALL-E continual | (b) | 60 000 soat | 3.8 | 0.508 | P-Flow Table 2 |
| P-Flow (2023) | (b) | **260 soat** / 2456 spiker | **2.6** | 0.544 | [NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Paper-Conference.pdf) |
| Voicebox (2023) | (b) | 60 000 soat | 1.9 | SIM-r 0.681 / SIM-o 0.662 | [2306.15687](https://arxiv.org/abs/2306.15687) |
| E2-TTS (2024) | (b) | 50 000 soat / 6736 spiker | 1.9 | SIM-o **0.708** (GT 0.695) | [2406.18009](https://arxiv.org/abs/2406.18009) |
| F5-TTS (2024) | (b) | ~95 000 soat (Emilia) | 2.42 | 0.66 | [2410.06885](https://arxiv.org/abs/2410.06885) |
| MaskGCT (2024) | (b) | 100 000 soat (Emilia) | 2.63 | SIM-o 0.687 | [2409.00750](https://arxiv.org/abs/2409.00750) |
| NaturalSpeech 3 (2024) | (a)+(b) gibrid | 200 000 soat | 1.81 | SIM-o 0.67 | [2403.03100](https://arxiv.org/abs/2403.03100) |
| VALL-E 2 (2024) | (b) | 50 000 soat (Libriheavy) | 1.5 | 0.643 (GT 0.779) | [2406.05370](https://arxiv.org/abs/2406.05370) |
| Seed-TTS (2024) | (b) | "oldingi eng kattadan bir necha martaba katta" | 2.249 (en) | 0.762 (en) | [2406.02430](https://arxiv.org/abs/2406.02430) |

> **Ogohlantirish — raqamlarni ko'r-ko'rona solishtirmang.** SIM/SECS qaysi speaker encoder bilan
> hisoblanganiga qarab o'zgaradi (WavLM-TDNN, ECAPA2, Resemblyzer — hammasi boshqa shkala),
> test to'plamlari ham turlicha. Voice Cloning Survey ([arXiv:2505.00579](https://arxiv.org/abs/2505.00579))
> buni aniq ta'kidlaydi: SECS'ni solishtirish qiyin, chunki *"the speaker encoder used for reporting
> results varies"*. Faqat bir maqola ichidagi qatorlar o'zaro solishtiriladi.

#### Uch o'qda taqqoslash

**Ovoz o'xshashligi (speaker similarity):** (b) aniq g'alaba qozonadi. Yuqoridagi P-Flow ablatsiyasi —
nazoratli dalil (0.373 → 0.544). E2-TTS SIM-o **0.708** bilan hatto **ground truth'dan (0.695) yuqori**
ball oldi.

**Prosodiya ko'chirish:** bu eng katta farq.
- (a) prosodiyani ko'chira **olmaydi** — GE2E loss uni ataylab yo'q qilgan (yuqoridagi 2-sabab).
  SV2TTS: *"some nuances, e.g. related to characteristic prosody, are lost"*.
- (b) prosodiyani **juda ko'p** ko'chiradi. VALL-E abstract'i buni yutuq deb yozadi:
  *"VALL-E could preserve the speaker's emotion and acoustic environment of the acoustic prompt"*.
  Lekin bu **ikki tomonli tig'** — referensda shovqin bo'lsa, shovqinni ham klonlaydi (5-bobga qarang).

**Ma'lumot talabi:** bu yerda manbalar **bir-biriga ZID**.
- Bir tomondan VALL-E 60 000 soat ishlatdi va bu "oldingi tizimlardan yuzlab marta katta" dedi.
  NaturalSpeech 3 skalalash jadvali ham 1K → 200K soatda Sim-O 0.64 → 0.73 ko'rsatdi.
- Ikkinchi tomondan **P-Flow 260 soatda** SECS 0.544 oldi (VALL-E'ning 60 000 soatdagi 0.580'iga
  yaqin) va ochiq yozdi: *"achieving high SECS does not require tens of thousands of hours of audio
  data, quantization, neural codec models, or large autoregressive models"*.
- **Bu to'g'ridan-to'g'ri zid.** Mening o'qishim bo'yicha kelishuv nuqtasi: (b) yo'lining **o'zi**
  (prompt-based conditioning) o'xshashlikni arzon beradi; 60 000 soat esa asosan **ravonlik,
  turli-tuman ovozlarni qoplash va robustlik** uchun kerak. 4-bobda bu batafsil.

---

# 2-BOB. Neural audio codec — (b) yo'lini mumkin qilgan komponent

## 2.1. "Audio token" nima va nega bu shunchalik muhim?

Matn tabiatan diskret: `["sa", "lom"]` — har biri lug'atdagi aniq raqam. Shuning uchun GPT
`softmax` orqali "keyingi token qaysi?" degan savolga javob bera oladi.

Tovush esa uzluksiz: 24 000 ta float soniya. Bunday narsaga til modeli qo'yish mumkin emas.

Neural audio codec shu muammoni yechadi:

```
  uzluksiz to'lqin                                       diskret tokenlar
  ──────────────────                                     ────────────────
  [0.013, -0.291, 0.44, ...]   ENCODER     QUANTIZER     [312, 88, 901, 45, ...]
   24 000 son/sek          ──────────>  ──────────────>   75 ta vektor/sek,
                            (320x        (eng yaqin        har biri 8 ta
                            siqish)       kodkitob          raqamga aylangan
                                          vektorini tanla)

                            <──────────────────────────
                                   DECODER
                            (tokenlardan tovush qaytariladi)
```

**Bir marta diskretlashtirgandan keyin tovush matnga aylanadi** — endi unga butun LLM
mashinasini (next-token prediction, cross-entropy, attention, sampling, hatto Qwen/LLaMA
checkpoint'larini) qo'llash mumkin. Mana shu — 2023-yildagi sakrash.

DAC maqolasi bu mantiqni birinchi abzatsida aytadi:
*"Language models have been successfully used to model natural signals... A key component of these
models is a high quality neural compression model that can compress high-dimensional natural signals
into lower dimensional discrete tokens."* ([arXiv:2306.06546](https://arxiv.org/abs/2306.06546))

## 2.2. Residual Vector Quantization (RVQ) — kodkitoblar portlashi muammosi

**Muammo:** bitta freym uchun 80 bit kerak bo'lsa, kodkitobda 2^80 ta vektor bo'lishi kerak.
SoundStream bu hisobni aniq keltiradi:

```
24 000 Hz / 320 stride = 75 freym/sek
6000 bit/sek / 75      = 80 bit/freym
bitta kvantizator      → N = 2^80 vektor →  "obviously unfeasible"
```

**Yechim: RVQ — kaskad.** Har bir kvantizator oldingisining **xatosini** (residual) kvantlaydi:

```
ALGORITM (SoundStream, arXiv:2107.03312):

  ŷ        := 0.0            # yakuniy taxmin
  residual := y              # encoder chiqishi

  for i = 1 .. Nq:
      ŷ        += Q_i(residual)     # i-kvantizator taxminini qo'sh
      residual -= Q_i(residual)     # qolgan xatoni hisobla

  return ŷ


VIZUAL:

   y  ─────────> [Q1] ──> qo'pol taxmin      ← bu "nima gapirilgan"ni belgilaydi
       xato  ───> [Q2] ──> + biroz aniqlik
       xato  ───> [Q3] ──> + yana aniqlik
          ...
       xato  ───> [Q8] ──> + eng nozik tafsilot   ← bu "qanday jaranglaydi"

   Nq=8 ta kvantizator × 2^(80/8) = 1024 vektor  →  AMALDA MUMKIN
```

Bu — **rassomning ishlash usuli**: avval qo'pol kontur, keyin asosiy ranglar, keyin soyalar,
oxirida nozik chiziqlar. Har bir qatlam oldingisining **yetishmagan qismini** to'ldiradi.

**Quantizer dropout va o'zgaruvchan bitrate:** o'qitishda har bir namuna uchun `n_q ~ U[1, Nq]`
tanlanadi — faqat birinchi `n_q` kvantizator ishlatiladi. Natijada **bitta model** barcha bitrate'larda
ishlaydi. DAC bir muhim tuzatish kiritdi: dropout'ni har bir namunaga emas, **batch'ning p=0.5
qismiga** qo'llash kerak, aks holda to'liq bandwidth'dagi sifat pasayadi.

**Kodkitob o'limi (codebook collapse):** VQ'ning klassik kasalligi — kodkitobning katta qismi
ishlatilmay qoladi. DAC ikki yechim topdi:
1. **Factorized codes** — qidirishni kichik o'lchovda (8d yoki 32d) qilib, embedding'ni yuqori
   o'lchovda (1024d) saqlash. *"principal component lookup maximizing data variance"*.
2. **L2-normalized codes** — Evklid masofasi o'rniga kosinus o'xshashlik.

WavTokenizer boshqa yo'l tutdi: **K-means bilan initsializatsiya (200 markaz) + "random awakening"**
(ishlatilmagan kodlarni batch'dan tasodifiy vektor bilan almashtirish).
CosyVoice 2 esa VQ'ni umuman tashlab **FSQ (Finite Scalar Quantization)**ga o'tdi va
**kodkitob ishlatilishini 23 % → 100 %** ko'tardi ([arXiv:2412.10117](https://arxiv.org/abs/2412.10117)).

## 2.3. Frame rate — nega bu TTS uchun hayot-mamot masalasi?

AR til modeli uchun **tokenlar soni = hamma narsa**: latency, attention xarajati (O(n²)),
o'qitish narxi, kontekst uzunligi.

```
  10 SEKUND NUTQ UCHUN NECHA TOKEN?

  EnCodec  75 Hz × 8 kodkitob  = 600 tok/sek →  6000 token   ████████████████████████
  DAC      86 Hz × 9 kodkitob  = 774 tok/sek →  7740 token   ███████████████████████████████
  X-codec2 50 Hz × 1 kodkitob  =  50 tok/sek →   500 token   ██
  WavTok.  40 Hz × 1 kodkitob  =  40 tok/sek →   400 token   ██
  CosyV.2  25 Hz × 1 (FSQ)     =  25 tok/sek →   250 token   █
  Mimi     12.5 Hz × 8         = 100 tok/sek →  1000 token   ████

  Taqqoslash uchun: MATN ≈ 3-4 token/so'z ≈ 10 token/sekund nutq
```

Moshi maqolasi buni ochiq aytadi: past frame rate *"is crucial to achieve the low latency of Moshi,
since generating one temporal frame of audio tokens... requires a full forward pass through the
Temporal Transformer"* ([arXiv:2410.00037](https://arxiv.org/abs/2410.00037)).

WavTokenizer esa sababni boshqa tomondan tushuntiradi: bitta kvantizator
*"enables direct autoregressive integration into language models without architectural workarounds"*
([arXiv:2408.16532](https://arxiv.org/abs/2408.16532)).

**Lekin savdolashuv bor:** kam token = kam bit = nozik tafsilotni tiklash qiyinroq.
WavTokenizer 40 token/sek'da UTMOS 3.60 oldi, 75 token/sek'da 4.05 — ya'ni tokenlarni
ikki baravar kamaytirganda sifat sezilarli pasaydi.

### Codec'lar jadvali (tekshirilgan raqamlar)

| Codec | Yil | Sample rate | **Frame rate** | Kodkitoblar | Kodkitob hajmi | Bitrate | Manba |
|---|---|---|---|---|---|---|---|
| **SoundStream** | 2021 | 24 kHz | **75 Hz** (stride 2·4·5·8=320) | Nq (tipik 8) | 1024 | 3 / 6 / 12 kbps | [2107.03312](https://arxiv.org/abs/2107.03312) |
| **EnCodec** | 2022 | 24 kHz / 48 kHz | **75 Hz** / 150 Hz | ≤32 (48kHz'da ≤16) | 1024 (10 bit) | 1.5 / 3 / 6 / 12 / 24 kbps | [2210.13438](https://arxiv.org/abs/2210.13438) |
| **DAC** | 2023 | 44.1 kHz | **86 Hz** (hop 512) | 9 | 1024 | 8 kbps (~91x siqish) | [2306.06546](https://arxiv.org/abs/2306.06546) |
| **SNAC** | 2024 | 24 / 32 / 44 kHz | **ierarxik** (qo'pol ~10 Hz) | 4 daraja, freymda [12,24,48,96] token | — | 0.98 / 1.9 / 2.6 kbps | [github/hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac) |
| **WavTokenizer** | 2024 | 24 kHz | **40 Hz** yoki **75 Hz** | **1** | 4096 (2^12) | 0.5 / 0.9 kbps | [2408.16532](https://arxiv.org/abs/2408.16532) |
| **Mimi** (Moshi) | 2024 | 24 kHz | **12.5 Hz** | 8 (**split RVQ**: 1 semantik + 7 akustik) | 2048 | 1.1 kbps | [2410.00037](https://arxiv.org/abs/2410.00037) |
| **X-codec2** (Llasa) | 2025 | — | **50 Hz** | **1** (FSQ) | **65 536** | — | [2502.04128](https://arxiv.org/abs/2502.04128) |
| **CosyVoice2 FSQ** | 2024 | — | **25 Hz** | **1** (FSQ) | — (100 % utilizatsiya) | — | [2412.10117](https://arxiv.org/abs/2412.10117) |
| **BiCodec** (Spark-TTS) | 2025 | — | 50 tok/sek semantik + **32 ta global token** | — | — | — | [2503.01710](https://arxiv.org/abs/2503.01710) |

> **SNAC'ning g'oyasi alohida e'tiborga arzidi:** qo'pol tokenlar **kamroq tez-tez** olinadi
> ("coarse tokens are sampled less frequently, covering a broader time span"). Ya'ni RVQ'ning
> darajalari bir xil vaqt rezolyutsiyasida bo'lishi shart emas. ~10 Hz qo'pol token + 2048
> kontekst oynasi = **~3 daqiqa** audio strukturasi.

> **Mimi'ning "split RVQ" hiylasi:** semantik ma'lumotni (WavLM distillatsiyasi orqali) oddiy
> RVQ'ning birinchi darajasiga tiqish rekonstruksiya sifatini buzadi, chunki yuqori kodkitoblar
> endi akustik cheklov ostida residual modellashi kerak. Yechim — semantik kodkitobni akustiklar
> bilan **ketma-ket emas, parallel** qo'yish: bu *"removes the constraint that acoustic information
> should be conserved"*.

## 2.4. "Birinchi kodkitob vs qolganlar" muammosi

RVQ bitta vaqt freymida **Nq ta token** beradi. Bu AR til modeli uchun katta muammo:

```
  RVQ chiqishi — MATRITSA, ketma-ketlik emas:

           t=1   t=2   t=3   t=4   ...
   Q1 →   [312] [ 88] [901] [ 45]      ← qo'pol / semantik: NIMA gapirilgan, RITM
   Q2 →   [ 77] [420] [ 12] [668]      ↑
   Q3 →   [201] [ 99] [333] [ 10]      │  nozik akustik tafsilot:
   ...                                  │  QANDAY jaranglaydi
   Q8 →   [ 55] [777] [  2] [444]      ↓

  Muammo: AR model chapdan o'ngga bitta ketma-ketlik kutadi.
  Yassilasak (flatten): 10 sekund = 6000 token. Juda uzun.
```

**Nega Q1 alohida?** Q1 **asos taxmin** — u gapning mazmunini va ritmini belgilaydi.
Qolganlari esa Q1 berilgan sharoitda ancha oldindan aytib bo'ladigan nozik tafsilotlar.

**Tarixiy 4 ta yechim:**

| Yechim | Kim | Qanday |
|---|---|---|
| **AR + NAR split** | VALL-E | Q1 — **AR** (ketma-ket), Q2..Q8 — **NAR** (parallel, bir o'tishda). *"NAR can reduce the time complexity from O(T) to O(1)"* |
| **Ierarxik ikki transformer** | Mimi/Moshi (**RQ-Transformer**), Fish-Speech (**DualAR**) | Katta **Temporal Transformer** vaqt bo'yicha + kichik **Depth Transformer** bir freymdagi K kodkitob bo'yicha |
| **Parallel masked decoding** | SoundStorm, MaskGCT | Darajalarni **qo'pol→nozik** tartibda, har bir daraja ichida parallel |
| **Muammoni umuman yo'q qilish** | WavTokenizer, X-codec2, CosyVoice2 FSQ | **Bitta kodkitob.** Matritsa yo'q, oddiy ketma-ketlik bor |

**VALL-E'ning AR/NAR bo'linishi — nega aynan shunday?** Maqola ikkita aniq sabab beradi:
1. **Nega Q1 AR bo'lishi kerak:** *"the rate of the generated speech should be consistent with the
   enrolled recording, and it is hard to train a length predictor"* — ya'ni **uzunlikni oldindan
   bilib bo'lmaydi**, shuning uchun AR `<eos>` chiqargancha davom etadi.
2. **Nega Q2..Q8 NAR bo'lishi mumkin:** ular bir-biriga shartli bog'liq emas darajada mustaqil;
   Q1 va prompt berilgan bo'lsa, parallel chiqarish sifatga zarar yetkazmaydi.

Yana bir muhim detal: **AR va NAR promptni boshqacha ishlatadi.** AR faqat promptning **1-kodkitobini**
prefix qiladi; NAR esa promptning **barcha 8 kodkitobini** oladi — chunki uning vazifasi aynan
akustik tafsilotni ko'chirish.

SoundStorm bu muammoni yana bir chiroyli trik bilan yengillashtirdi: bidirectional Conformer
*"sums up the embeddings of the tokens corresponding to the same SoundStream frame, such that the
internal sequence length for the self-attention is identical to the number of SoundStream frames,
and independent of the number of quantizers"*. Ya'ni attention uzunligi **kvantizator sonidan
mustaqil** bo'ladi.

VALL-E 2 esa **Grouped Code Modeling** qo'shdi: kodlarni G ta guruhga bo'lib, har bir guruhni
bitta freym deb qarash. G=4'da ketma-ketlik 4 marta qisqaradi va bu *"not only accelerates inference
by reducing the sequence length but also improves performance by mitigating the long context
modeling problem"*.

---

# 3-BOB. Generativ "dvigatel"ning uch avlodi

## 3.1. 1-avlod: Autoregressive codec LM (GPT uslubi)

**Mexanizm.** To'liq GPT. Bitta Transformer, chapdan o'ngga, bir vaqtda bitta token.
Loss — cross-entropy. Sampling — nucleus/top-p.

```
  [matn tokenlari][prompt audio tokenlari] → t1
  [matn tokenlari][prompt audio tokenlari][t1] → t2
  [matn tokenlari][prompt audio tokenlari][t1][t2] → t3
  ...
  ... → <eos>   ← UZUNLIKNI MODELNING O'ZI QAROR QILADI
```

**Bu avlod a'zolari:** VALL-E / VALL-E 2, XTTS, Fish-Speech (DualAR), Llasa (LLaMA asosida),
Spark-TTS (Qwen2.5-0.5B asosida), CosyVoice 2 (LM qismi).

**Kuchli tomoni — bu avlodning yagona strukturaviy ustunligi: uzunlik.** Duration model
**kerak emas**, chunki model `<eos>`ni o'zi chiqaradi. Tabiiy ritm, tabiiy pauza.
Shu bilan birga **streaming** oson: Fish-Speech first-packet latency **150 ms**;
CosyVoice 2 ham streaming'ni shu tufayli qiladi.

**Zaif tomoni — beqarorlik.** Bu avlodning kasalligi chuqur va yaxshi hujjatlashtirilgan (5-bobda batafsil):
VALL-E o'zi tan oladi *"some words may be unclear, missed, or duplicated... because the
phoneme-to-acoustic language part is an autoregressive model, in which disordered attention
alignments exist"*.

**Bu avlodda LLM checkpoint'laridan foydalanish mumkin** — Llasa LLaMA'ni, Spark-TTS Qwen2.5-0.5B'ni
lug'atini kengaytirib ishlatadi. Bu katta amaliy afzallik (matn tushunish allaqachon tayyor).

## 3.2. 2-avlod: Non-autoregressive flow matching / diffusion

### Flow matching — oddiy tilda

Diffusion'ni tasavvur qiling: tumanda tog'dan tushish. Har qadamda atrofga qaraysan, oz-oz
pastga qadam bosasan. 150–1000 qadam kerak.

Flow matching (optimal transport yo'li bilan) boshqacha: sen **har bir nuqtada qaysi yo'nalishda
va qanday tezlikda harakat qilish kerakligini** o'rganasan — va bu yo'l **to'g'ri chiziq** bo'ladi.
To'g'ri chiziqda katta qadam tashlash mumkin. **16 qadam, hatto 10 qadam** yetadi.

```
  DIFFUSION:                      FLOW MATCHING (OT yo'li):

  shovqin                         shovqin
    ·                                ·
     \_/\_/\_/\                       \
      \_/\_/\_/\                       \      ← o'rganilgan tezlik maydoni v_t(x)
       \_/\_/\_/\                       \        to'g'ri va qisqa
        ·  (150+ qadam)                  ·  (16 qadam)
      nutq                             nutq
```

Formula sodda: model har bir (vaqt, nuqta) juftligi uchun **tezlik vektorini** taxmin qiladi va
haqiqiy tezlikka yaqinlashadi:
```
L_CFM(θ) = E_{t, q(x1), p_t(x|x1)}  ‖ u_t(x | x1)  −  v_t(x; θ) ‖²
                                      ^^^^^^^^^^^      ^^^^^^^^^
                                      haqiqiy tezlik    model taxmini
```
Voicebox: OT yo'li *"points move linearly between distributions with constant speed/direction"*,
natijada **<16 NFE** (ba'zan <10) yetarli va bu VALL-E'dan **20 martagacha tez**
(10 sekund audio uchun 0.31 s vs 6.2 s).

### "Infilling" formulasi — bu bobning eng muhim qismi

Mana shu yerda haqiqiy nozik g'oya bor: **speaker encoder ham, duration model ham kerak emas**
qiladigan narsa.

**Vazifani qayta ta'riflash.** TTS'ni "matndan nutq yasash" deb emas, **"nutqning o'chirilgan
qismini tiklash"** deb qo'yamiz:

```
  O'QITISHDA (E2-TTS):

  Haqiqiy audio:   ████████████████████████████████████████████
  Tasodifiy maska: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                   ^ ko'rinadi  ^ 70–100 % O'CHIRILGAN

  Model o'rganadi:  P( m⊙ŝ  |  (1−m)⊙ŝ ,  ŷ )
                      ^^^^      ^^^^^^^^    ^
                      tiklash   qolgan      butun gapning matni
                      kerak     audio

  Matn esa mel uzunligiga TO'LDIRUVCHI token bilan cho'ziladi:
  ŷ = (c1, c2, ..., cm, ⟨F⟩, ⟨F⟩, ⟨F⟩, ..., ⟨F⟩)
                        ^^^^^^^^^^^^^^^^^^^^^^^ filler
  → matn uzunligi == mel freymlar soni → ALIGNMENT MUAMMOSI YO'Q
```

```
  INFERENCE'DA (xuddi shu model, hiyla bilan):

  "Ko'rinadigan kontekst" deb REFERENS AUDIOni beramiz:
  ┌─────────────────────┬────────────────────────────────────┐
  │ referens mel (bor)  │  shovqin → tiklanishi kerak        │
  └─────────────────────┴────────────────────────────────────┘
  Matn: [referens transkripsiyasi] + [yangi matn] + ⟨F⟩⟨F⟩...
         ^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^
         chap qism nimaga mos         o'ng qism nima yozilishi
         kelishini aytadi             kerakligini aytadi

  Model "yetishmagan qismni to'ldiradi" — va to'ldirish uchun
  chapdagi audioning ovoziga MOS KELISHI kerak, chunki o'qitishda
  u har doim BIR GAPNING ichini to'ldirgan.
```

**Nega speaker encoder kerak emas?** Chunki referens mel **jismonan shu yerda** —
Transformer self-attention orqali uni to'g'ridan-to'g'ri ko'radi. E2-TTS'ning o'z izohi:
*"The reference mel-spectrogram prefix implicitly encodes speaker characteristics. The transformer
learns to generate acoustically-consistent speech matching that speaker without explicit speaker
embedding extraction."*

**Nega duration model kerak emas?** Chunki matn **filler token bilan to'liq uzunlikka cho'zilgan**.
Model **umumiy uzunlikni biladi** va har bir so'zga qancha joy berishni o'zi hal qiladi.
F5-TTS buni shunday izohlaydi: *"the model learns to assign the length of each word or phoneme
according to the given total sequence length, resulting in improved prosody and rhythm."*
Umumiy uzunlik esa inference'da **belgilar sonining nisbati** bilan taxmin qilinadi
(`len(yangi matn) / len(referens matn) × len(referens audio)`). Shu darajada sodda.

> **Muhim farq — Voicebox vs E2/F5.** Voicebox ham infilling'dan foydalanadi, lekin u **hali ham
> alohida duration model** ishlatadi (phone-darajali davomiylik uchun flow-matching yoki L1
> regressiya; default — regressiya). E2-TTS va F5-TTS esa duration model'ni **butunlay yo'q qildi**.
> Shuning uchun "flow matching = duration model yo'q" degan gap **noto'g'ri** — bu E2/F5 oilasiga xos.

**Sway Sampling (F5-TTS).** Flow qadamlarini bir tekis emas, **boshiga to'playdi**:
```
f_sway(u, s) = u + s · ( cos(π/2 · u) − 1 + u ),    s ∈ [−1, 2/(π−2)]
```
Sabab: *"alignment between speech and text will be determined based on the first few generated
results"* — ya'ni matn bilan tovush mosligi **ilk qadamlarda** qaror topadi, demak o'sha joyga
ko'proq qadam sarflash kerak. Default `NFE=32`, `s = −1`.

## 3.3. 3-avlod: Masked generative / parallel iterativ dekodlash

### Mexanizm — krossvord analogiyasi

AR model krossvordni **chapdan o'ngga, harf-harf** yozadi va orqaga qaytolmaydi.
Masked generative model boshqacha ishlaydi:

```
  1-qadam:  hammasi bo'sh         [?][?][?][?][?][?][?][?][?][?][?][?]
            hammasini bir yo'la taxmin qil, faqat ISHONCHLI larini qoldir
                                  [?][?][ C][?][?][?][ A][?][?][?][?][?]

  2-qadam:  qolganini yana taxmin qil (endi C va A ni ko'rib turib)
                                  [?][ B][C][?][?][ T][A][?][?][ K][?][?]

  3-qadam:                        [ Z][B][C][ M][ N][T][A][ R][ Q][K][?][ D]

  ...  ~16–50 qadamdan keyin hammasi to'ldi.
```

Tanlov **confidence-based**: SoundStorm *"samples candidates for the masked positions, retaining
p_i of them based on confidence scores"*, p_i kosinus jadval bo'yicha o'sib boradi.
Oxirgi iteratsiyada greedy.

RVQ darajalari **qo'pol→nozik** tartibda: *"only proceeding to level q+1 when all tokens for
levels 1..q have been sampled"*, chunki *"tokens of finer levels are responsible for local, fine
acoustic details and can thus be sampled in parallel without a loss of audio quality"*.

**Tezlik — bu avlodning asosiy yutug'i:**
```
SoundStorm:  (16, 1, 1, ..., 1) iteratsiya  =  JAMI 27 forward pass
             30 sekund audio = 18 000 token
             TPU-v4'da 0.5 sekund  →  AudioLM'dan ~100× (ikki tartib) tez
```

**Prompt qanday ishlaydi?** Bu yerda ham (b) yo'li: **speaker embedding YO'Q**.
SoundStorm'da o'qitishda tasodifiy `t` tanlanib, `t`dan oldingi tokenlar **hech qachon
maskalanmaydi**. Inference'da promptning (~3 sek) tokenlari boshdan oxirgacha **maskalanmagan
holda turadi** va model qolganini ularga mos qilib to'ldiradi.

MaskGCT xuddi shu printsipni ikki bosqichli to'liq-NAR quvurga chiqardi:
```
  matn ──> [T2S: masked transformer] ──> semantik tokenlar (W2v-BERT 2.0 VQ)
                                               │
           prompt semantik tokenlar ───────────┤ (maskalanmagan prefix)
                                               v
                                        [S2A: masked transformer] ──> akustik tokenlar (12-qatlamli RVQ)
                                               │
           prompt akustik tokenlar ────────────┘
                                               v
                                            codec decoder ──> tovush
```
va **alignment ham, duration prediction ham kerak emas**: *"no text-speech alignment supervision
and phone-level duration prediction"*. Qadamlar: T2S uchun **50**, S2A uchun
`[40, 16, 1, 1, ... ]`. Natija: LibriSpeech SIM-o 0.687 / WER 2.634,
SMOS va CMOS **ground truth darajasida**.

NaturalSpeech 3 ham masked diskret diffusion ishlatadi, lekin **atributlarga bo'lib**: har bir
atribut (duration, prosodiya, kontent, akustik tafsilot) uchun **4 iteratsiya** —
classifier-free guidance bilan jami **60 forward pass**.

## 3.4. UCH AVLOD — TAQQOSLASH JADVALI

| | **1-avlod: AR codec LM** | **2-avlod: NAR flow matching / diffusion** | **3-avlod: Masked generative** |
|---|---|---|---|
| **Vakillar** | VALL-E, VALL-E 2, XTTS, Fish-Speech, Llasa, Spark-TTS, CosyVoice2 (LM) | Voicebox, E2-TTS, **F5-TTS**, CosyVoice2 (flow decoder), Seed-TTS_DiT | SoundStorm, **MaskGCT**, NaturalSpeech 3 |
| **Mexanizm** | Keyingi token = softmax, chapdan o'ngga | Shovqindan nutqqa ODE bo'ylab integrallash | Hammasi maskalangandan boshlab iterativ to'ldirish |
| **Nechta o'tish (10 s nutq)** | ~600–6000 (har token = 1 o'tish) | **16–32 NFE** | **27** (SoundStorm) / ~50+ (MaskGCT) / 60 (NS3) |
| **Tezlik** | Sekin. VALL-E: 10 s audio uchun ~6.2 s | Tez. Voicebox VALL-E'dan **20×**; F5 **RTF 0.15** (16 NFE) | Juda tez. SoundStorm: 30 s audio / 0.5 s = **~100×** AudioLM'dan |
| **Uzunlik (duration)** | **Model o'zi hal qiladi** (`<eos>`). Duration model shart emas | Voicebox: **duration model KERAK**. E2/F5: **kerak emas** (filler token + belgi nisbati) | **Kerak emas** (implicit, mask orqali) |
| **Barqarorlik / hallucination** | **Eng yomon.** So'z tushib qoladi, takrorlanadi, "cheksiz tsikl". VALL-E hard sentences'da **68 % xato** | **Yaxshi.** WER past (Voicebox 1.9, E2 1.9, F5 2.42). Lekin E2-TTS test'ning **7 %**ida alignment buziladi | **Yaxshi.** MaskGCT WER 2.63; NS3 WER 1.81 |
| **Streaming** | **Oson** — token-token. Fish-Speech first-packet **150 ms**; CosyVoice2 streaming | Qiyin — butun ketma-ketlik bir yo'la hisoblanadi. CosyVoice2 **chunk-aware** flow bilan yechdi | Qiyin (bir yo'la butun bloklar) |
| **Prosodiya** | Juda ekspressiv, lekin beqaror. Seed-TTS: AR'da "consistency bias" — bir gap ichida prosodiya juda bir xil | Barqaror, ba'zan "tekisroq" | Barqaror |
| **LLM checkpointdan foydalanish** | **Ha** — Llasa (LLaMA), Spark-TTS (Qwen2.5) | Yo'q | Yo'q |
| **Codec kerakmi** | **Ha, diskret tokenlar majburiy** | **Yo'q** — ko'pincha to'g'ridan-to'g'ri **mel** (uzluksiz) | Ha, diskret |
| **Eng yaxshi SIM-o** | VALL-E 2: 0.643 | **E2-TTS: 0.708** (GT 0.695) | MaskGCT: 0.687 |

## 3.5. Vocoder / decoder qayerda turadi va hali ham kerakmi?

Javob: **bog'liq — qaysi vakilligi ishlatilganiga.**

```
  AGAR MODEL DISKRET CODEC TOKEN CHIQARSA (1 va 3 avlod):

    tokenlar ──> [CODEC DECODER] ──> to'lqin
                  ^^^^^^^^^^^^^
                  alohida vocoder KERAK EMAS — codec o'zining
                  decoderi bilan keladi (EnCodec, DAC, Mimi, X-codec2).
                  Lekin istasang codec decoder o'rniga Vocos qo'yish mumkin.

  AGAR MODEL MEL-SPEKTROGRAMMA CHIQARSA (2-avlod, E2/F5/Voicebox):

    mel ──> [VOCODER] ──> to'lqin
             ^^^^^^^
             MAJBURIY. F5-TTS repo'si Vocos (vocos-mel-24khz)
             va BigVGAN variantlarini beradi.

  ARALASH (XTTS, Fish-Speech, CosyVoice2):

    LM latent / tokenlar ──> [HiFi-GAN / FF-GAN / flow+vocoder] ──> to'lqin
    (XTTS: HiFi-GAN VQ-VAE kodlaridan emas, GPT-2 LATENT fazosidan tiklaydi —
     26M parametr, speaker embedding har bir upsampling qatlamida qo'shiladi)
```

| Vocoder | Yil | Asosiy g'oya | Manba |
|---|---|---|---|
| **HiFi-GAN** | 2020 | GAN, transposed convolution, multi-receptive field | XTTS, OpenVoice, YourTTS'da ishlatiladi |
| **BigVGAN** | 2022 | **snake (periodik) aktivatsiya** + anti-aliased multi-periodicity composition (AMP); **112M parametr**; LibriTTS'da o'qitilgan; *"universal vocoder that generalizes well for various out-of-distribution scenarios without fine-tuning"* — ko'rilmagan spiker, til, qo'shiq, musiqaga ham ishlaydi | [2206.04658](https://arxiv.org/abs/2206.04658) |
| **Vocos** | 2023 | Transposed convolution o'rniga **to'g'ridan-to'g'ri Fourier koeffitsiyentlarini** chiqarib ISTFT qiladi → *"an order of magnitude increase in speed"* | [2306.00814](https://arxiv.org/abs/2306.00814) |
| **FF-GAN** (Firefly) | 2024 | HiFi-GAN'ning MRF moduli o'rniga ParallelBlock + GFSQ | [2411.01156](https://arxiv.org/abs/2411.01156) |

> **Amaliy xulosa:** vocoder TTS quvurining eng "yechilgan" qismi. BigVGAN/Vocos tayyor holda
> olib ishlatiladi — bu qismni o'zi o'qitish kerak emas.

---

# 4-BOB. NEGA ULKAN KO'P-SPIKERLI DATASET MUQARRAR?

Bu — foydalanuvchining asosiy savoli. Javob konseptual jihatdan oddiy, lekin dalillar nozik.

## 4.1. Asosiy sabab: model BITTA ovozni emas, OVOZLAR FAZOSINI o'rganadi

```
  ODDIY (single-speaker) TTS:                ZERO-SHOT TTS:

      ●                                            ●  ●     ●
     bitta                                       ●    ●   ●    ●
     nuqtani                                    ●   ●  ● ●  ●    ●
     yodlash                                      ●    ●   ●   ●
                                                ●  ●  ●  ●   ●
   "Dilnozaning ovozi"                        ●    ●    ●    ●  ●
   → 5-20 soat o'sha odam                       ●   ●  ●   ●
                                             BUTUN FAZONI o'rganish
                                             → yangi nuqta (★) kelganda
                                               interpolatsiya qilish

                                                     ★ ← senning 10 sekundingiz
```

Zero-shot bo'lishi uchun model **"men hali ko'rmagan ovoz ham shu fazoning ichida"** deb
ayta olishi kerak. Fazoning chegaralari, o'lchovlari, ichki strukturasi faqat **ko'p turli
odamlarni ko'rish** bilan o'rganiladi. Bu printsipial — matematik emas, statistik majburiyat:
**model o'zi ko'rmagan ovoz turini ekstrapolyatsiya qilolmaydi**.

Voice Cloning Survey zero-shot ta'rifini ham shu nuqtadan beradi: uning farqi shunda ki
*"you do not need to fine-tune the TTS model"* — ya'ni butun og'irlik **oldindan o'qitishga**
tushadi. Agar oldindan o'qitish fazoni qoplamagan bo'lsa, tushadigan joy qolmaydi.

## 4.2. Dalillar: miqdor oshganda nima bo'ladi?

### Dalil 1 — NaturalSpeech 3, soat bo'yicha skalalash (eng toza jadval)

| O'qitish soati | **Sim-O ↑** | WER ↓ |
|---|---|---|
| 1 000 | 0.64 | 3.94 |
| 60 000 (LibriLight) | 0.72 | 3.03 |
| 200 000 (ichki) | **0.73** | **2.11** |

([arXiv:2403.03100](https://arxiv.org/abs/2403.03100), Table 7)

**Diqqat qiling:** 1K → 60K (60 marta ko'p) Sim-O'ni **+0.08** ko'tardi.
60K → 200K (3.3 marta ko'p) esa faqat **+0.01**. Ya'ni **o'xshashlik 60K soatda to'yinadi**.
WER esa to'yinmaydi — oxirgacha yaxshilanib boradi (3.03 → 2.11).

### Dalil 2 — Llasa, ma'lumot vs model hajmi (bu ikkisini ajratgan yagona jadval)

**Ma'lumot ko'paytirsak** (1B parametr qotirilgan):
| Soat | Spiker o'xshashligi (test-en) | WER (test-en) |
|---|---|---|
| 80 000 | 0.541 | 3.71 |
| 160 000 | 0.563 | 3.60 |
| 250 000 | **0.572** | **3.22** |

**Model ko'paytirsak** (250K soat qotirilgan):
| Parametr | Spiker o'xshashligi (test-en) | WER (test-hard) |
|---|---|---|
| 1B | 0.572 | 12.13 |
| 3B | 0.579 | 13.37 |
| 8B | **0.574** | **11.09** |

([arXiv:2502.04128](https://arxiv.org/abs/2502.04128))

> **Bu ikki jadval juda muhim xulosa beradi:** ma'lumotni 3 marta oshirganda o'xshashlik
> **0.541 → 0.572** (+0.031) o'sdi; modelni **8 marta** oshirganda esa
> **0.572 → 0.574** (+0.002) — ya'ni **deyarli hech narsa**.
> Llasa'ning o'zi ham aytadi: chet so'zlar va qo'shma otlar *"primarily benefit from increased
> training data rather than model scaling"*.
>
> **Ovoz o'xshashligi uchun katta GPU emas, katta va xilma-xil DATASET kerak.**

### Dalil 3 — SV2TTS: spiker soni (soat emas!) bo'yicha ablatsiya

Bu yo'l (a) uchun, lekin eng toza spiker-soni dalili:

| Speaker encoder o'qitilgan spiker soni | Sintez ovozning SV-EER ↓ |
|---|---|
| 1 200 | 16.60 % |
| 8 400 | 10.14 % |
| 18 000 | **5.08 %** |

Vazifa va arxitektura o'zgarmagan — faqat **spiker soni**. Xato 3 baravardan ko'p pasaydi.

### Dalil 4 — YourTTS Exp 1 vs Exp 4 (eng nozik dalil)

| Tajriba | Spiker soni | **VCTK SECS** (in-domain) | **LibriTTS SECS** (out-of-domain) |
|---|---|---|---|
| Exp 1: faqat VCTK | 98 | **0.864** | 0.754 |
| Exp 2: + portugalcha | ~103 | 0.857 | 0.762 |
| Exp 3: + fransuzcha | ~108 | 0.851 | 0.761 |
| Exp 4: **+ LibriTTS 1151 spiker** | ~1 250 | 0.843 | **0.856** |

**Mana bu — butun hikoyaning mikrokosmi.** 1151 spiker qo'shilganda:
- in-domain (VCTK) o'xshashligi **pasaydi** (0.864 → 0.843)
- out-of-domain (LibriTTS) o'xshashligi **sakrab ketdi** (0.754 → 0.856, **+0.10**)

Ya'ni **ko'p spiker = yaxshiroq umumlashtirish**, lekin o'z domenidagi aniqlikdan biroz qurbonlik.
Zero-shot uchun kerak bo'lgan narsa — aynan **umumlashtirish**.

YourTTS buni ochiq yozadi: *"ZS-TTS models still require a considerable amount of speakers for
training, making it difficult to obtain high-quality models in low-resource languages."*

### Dalil 5 — Ma'lumot SIFATI/XILMA-XILLIGI ham soatdan muhim

Emilia maqolasi ([arXiv:2407.05361](https://arxiv.org/abs/2407.05361)) bir xil model arxitekturasini
**audiokniga** (MLS) va **"in-the-wild"** (Emilia) ma'lumotida o'qitib taqqosladi:

| Model | Dataset | WER ↓ | SIM-O ↑ |
|---|---|---|---|
| AR | MLS (audiokniga) | 7.7 | 0.587 |
| AR | **Emilia (in-the-wild)** | **6.6** | **0.618** |
| NAR | MLS | 8.2 | 0.528 |
| NAR | **Emilia** | **7.4** | **0.601** |

Ya'ni **turli-tuman, spontan, "nopok" ma'lumot studiya audiokniga ma'lumotidan yaxshiroq**
zero-shot beradi. Chunki zero-shot'da sen ko'rishni kutayotgan referenslar ham studiya emas.

Shu bilan bir qatorda, yangi preprint
([arXiv:2603.10904](https://arxiv.org/html/2603.10904), Purwar & Choudhary, 2026-03-11)
LoRA fine-tuning kontekstida xuddi shu xulosaga keldi: energiya standart chetlanishi **>13 dB**
bo'lgan spikerlarda DNS-MOS **+0.42** yaxshilandi, **<10 dB** bo'lganlarda esa **−0.41** yomonlashdi —
**mavjud soatdan qat'i nazar**. Hamda faqat HiFi-TTS'da o'qitilgan ko'p-spikerli model
ko'rilmagan LibriHeavy spikerlarida **+0.29 DNS-MOS** oldi.
> ⚠️ Bu kichik, kam ma'lum preprint (6 spiker, Qwen-0.5B) — asosiy dalil sifatida emas,
> yo'nalishni tasdiqlovchi ikkinchi darajali dalil sifatida o'qing.

## 4.3. ZID DALIL: P-Flow — 260 soat yetarli bo'lishi mumkin

Haqiqatni to'liq aytish kerak: **manbalar bu yerda bir-biriga zid.**

P-Flow (NVIDIA, NeurIPS 2023) LibriTTS'ning **faqat 260 soatlik** qismida (2456 spiker)
o'qitildi va quyidagini oldi:

| Model | Soat | WER ↓ | SECS ↑ | Latency ↓ |
|---|---|---|---|---|
| Ground truth (HiFi-GAN) | — | 2.4 | 0.64 | — |
| YourTTS | 500+ | 7.7 | 0.337 | — |
| VALL-E | **60 000** | 5.9 | **0.580** | 2.515 s |
| VALL-E continual | 60 000 | 3.8 | 0.508 | 2.515 s |
| **P-Flow** | **260** | **2.6** | **0.544** | **0.115 s** |

P-Flow'ning matn encoder'i faqat **3M parametr**. Va maqola to'g'ridan-to'g'ri aytadi:

> *"achieving high SECS does not require tens of thousands of hours of audio data, quantization,
> neural codec models, or large autoregressive models... we believe that prompting is one of the
> most important factors in obtaining high SECS."*

**Bu VALL-E/NaturalSpeech 3 ruhiga ochiq zid.** Ikkalasini ham hisobga olish kerak.

### Mening o'qishim (bu — xulosa, nashr etilgan da'vo emas)

Yuqoridagi barcha jadvallarni birga qo'yganda quyidagi manzara chiqadi:

```
  NIMA UCHUN NIMA KERAK:

  O'XSHASHLIK (SIM / SECS)
    ├─ Eng kuchli omil:  CONDITIONING USULI (prompt vs embedding)
    │                     → P-Flow: 0.373 → 0.544, bir xil ma'lumotda
    ├─ Ikkinchi omil:    SPIKERLAR XILMA-XILLIGI
    │                     → SV2TTS: 1.2K→18K spiker, EER 16.6%→5.1%
    │                     → YourTTS: +1151 spiker, OOD SECS 0.754→0.856
    └─ 60K soatdan keyin TO'YINADI (NS3: 0.72 → 0.73)

  RAVONLIK / TUSHUNARLILIK (WER)
    ├─ SOATLAR bilan oxirigacha yaxshilanadi
    │   → NS3: 3.94 → 3.03 → 2.11
    │   → Llasa: 3.71 → 3.60 → 3.22
    └─ TRANSKRIPSIYA SIFATI juda muhim
        → P-Flow 260 soatda WER 2.6 oldi, sababi: "training on the more
          accurately transcribed LibriTTS dataset"

  MODEL HAJMI (parametrlar)
    └─ O'xshashlikka DEYARLI TA'SIR QILMAYDI (Llasa: 1B→8B, 0.572→0.574)
```

**Ya'ni: soatlar talaffuzni sotib oladi, spikerlar xilma-xilligi o'xshashlikni sotib oladi.**

## 4.4. "Spiker soni" vs "soat soni" — qaysi muhimroq? To'g'ridan-to'g'ri dalil

**Rostini aytish kerak: jami soatni qotirib, spiker sonini o'zgartirgan toza nazoratli
tajriba zero-shot TTS sintezatori uchun MA'LUMOT TOPILMADI.** Men izlagan manbalarda
bunday ablatsiya yo'q. Mavjud skalalash tajribalarida (NS3, Llasa, Emilia) soat va spiker
**birga** o'sadi — ajratilmagan.

Bilvosita dalillar esa hammasi bir yo'nalishni ko'rsatadi:

| Dalil | Nimani o'zgartirgan | Natija |
|---|---|---|
| SV2TTS Table 5 | **faqat spiker soni** (encoder) | 1.2K→18K: EER 16.6%→5.1% |
| YourTTS Exp1→Exp4 | spiker + soat (aralash) | OOD SECS +0.10 |
| P-Flow | **260 soat, lekin 2456 spiker** | SECS 0.544 — 60 000 soatli VALL-E'ga yaqin |
| YourTTS | **500+ soat, ~1250 spiker**, lekin embedding yo'li | SECS 0.337 — ancha past |
| arXiv:2603.10904 | akustik xilma-xillik | diversity > quantity |

> **P-Flow vs YourTTS taqqoslashi diqqatga sazovor:** P-Flow 260 soat / 2456 spikerda 0.544,
> YourTTS 500+ soat / ~1250 spikerda 0.337 oldi. **Kamroq soat, ko'proq spiker, va (b) yo'li**
> = ikki baravar yaxshi. Lekin arxitekturalari ham farq qiladi, shuning uchun bu nazoratli
> taqqoslash emas.

**Amaliy xulosa:** agar sening budjetingda tanlov bo'lsa — **200 soat × 1000 spiker**
**2000 soat × 50 spikerdan** zero-shot uchun deyarli aniq yaxshiroq. Lekin buni raqam bilan
tasdiqlaydigan nashr topilmadi.

---

# 5-BOB. Zero-shot klonlashning MA'LUM NUQSONLARI

## 5.1. AR modellar: hallucination, takrorlanish, so'z tushib qolishi

Bu eng yaxshi hujjatlashtirilgan nuqson.

**Ildiz sabab (RALL-E, [arXiv:2404.03204](https://arxiv.org/abs/2404.03204)) — uchta mexanizm:**

1. **Alignment yo'q:** *"Since there is no strict alignment between text and speech, the models
   can omit or repeat words in the input text."*
2. **Implicit attention noto'g'ri yo'naladi:** *"the speech token attends to all phonemes in the
   AR Transformer of VALL-E, the alignment between the phonemes and the speech tokens is implicitly
   modeled by the self-attention... This can be imprecise and causes errors like word omissions
   or hallucinations."*
3. **Davomiylik nazorat qilinmaydi:** model *"directly generates speech from phonemes with no
   restriction on the prosody, e.g. pitch, duration"* → beqaror ritm.

VALL-E o'zi ham tan oladi: *"some words may be unclear, missed, or duplicated... because the
phoneme-to-acoustic language part is an autoregressive model, in which disordered attention
alignments exist."*

**RAQAMLAR — bu qanchalik yomon?**

RALL-E 50 ta ataylab qiyin qilingan gapda sinadi:

| Model | Xato darajasi | Tafsilot |
|---|---|---|
| **VALL-E** | **68 %** | 10 noto'g'ri talaffuz, **19 so'z tushib qolgan**, **8 takrorlanish**, **7 hallucination** |
| RALL-E | 4 % | faqat 2 noto'g'ri talaffuz |

Oddiy LibriSpeech test-clean'da ham: VALL-E WER **5.6 %** → RALL-E **2.5 %**
(re-ranking bilan 1.7 % → 1.0 %).

RALL-E alohida eslatadi: *"For extremely short sentences, e.g. a single alphabet 'A', VALL-E
usually synthesizes words that don't exist in the input sentence."*

**"Cheksiz tsikl" muammosi (VALL-E 2, [arXiv:2406.05370](https://arxiv.org/abs/2406.05370)):**
*"The random sampling used during inference can lead to instability in output, while nucleus
sampling with a small top-p value may cause an infinite loop issue."*

Ya'ni sampling'ning **ikki uchi ham yomon**: tasodifiy — beqaror; juda qotirilgan — tsikl.

**Yechim — Repetition Aware Sampling:** nucleus sampling qilinadi, keyin oxirgi **K=10** token
oynasida takrorlanish nisbati hisoblanadi; agar **tr=0.1**'dan oshsa, token tasodifiy sampling
bilan almashtiriladi. Natija: WER **3.1 → 1.5**, va cheksiz tsikl yo'q bo'ldi.

**Amaliy e'tibor:** VALL-E ilgari bu muammoni **ko'p marta sample qilib, keyin saralash** bilan
yengardi — bu *"increasing the computational cost"*. Ya'ni AR modelni amalda ishlatganda
**bir necha marta generatsiya qilib eng yaxshisini tanlash** kerak bo'lishi mumkin.

## 5.2. NAR modellar ham immunitetga ega emas: alignment buzilishi

E2-TTS (flow matching, AR emas) butunlay boshqa kasallikni ko'rsatdi:
F5-TTS maqolasiga ko'ra E2-TTS *"consistent failure... on a 7% of the test set (WER>>50%)
all along the training process"*. Ya'ni test to'plamining **7 %**ida model **mutlaqo** buzilib
ketadi, va bu *"could not simply be solved with re-ranking"*.

Sabab: *"deep entanglement of semantic and acoustic features"* — matn va tovush xususiyatlari
chatishib ketgan. F5-TTS buni ConvNeXt bilan matnni oldindan tayyorlab yechdi.

> **Xulosa: 3 ta avlodning hech biri 100 % ishonchli emas.** AR — so'z tushiradi/takrorlaydi;
> NAR — vaqti-vaqti bilan butunlay buziladi; ikkisi ham orqa fonda tekshiruv (ASR bilan
> WER o'lchash) talab qiladi.

## 5.3. Davomiylik (duration) beqarorligi

- **AR:** uzunlikni model o'zi hal qiladi → tabiiy, lekin nazorat yo'q. RALL-E'ning yechimi:
  **duration-guided masked attention** (har bir nutq tokeni faqat o'ziga mos fonema
  atrofidagi oynaga qaraydi, `k=1`'da 3 fonema) + **D-qadamda majburan to'xtatish**.
- **Voicebox:** alohida duration model kerak → bu qo'shimcha xato manbai.
- **E2/F5:** duration **belgilar nisbati** bilan taxmin qilinadi → agar referens matni va
  yangi matn tilining belgi-ovoz nisbati boshqa bo'lsa (masalan lotin o'zbek vs ingliz),
  tezlik noto'g'ri chiqishi mumkin. ⚠️ Bu — o'zbek tili uchun amaliy xavf; F5 maqolasida bu
  aniq aytilmagan, bu mexanizmdan kelib chiqadigan xulosa.
- **Seed-TTS_DiT:** o'qitishda **umumiy davomiylik** beriladi va model fonemalar bo'yicha
  *"dynamically adjust[s] the duration of each phoneme"*.

## 5.4. Referensdan AKSENT oqib o'tishi

**Yo'l (a)da:** SV2TTS speaker encoder faqat Shimoliy Amerika aksentida o'qitilgan edi, natijada
britancha VCTK spikerlarida sifat pasaydi. Ya'ni **embedding fazosi ko'rmagan aksentni yomon
kodlaydi**.

**Yo'l (b)da — kross-lingual holat ancha yomon.** Prompt audio tilidagi aksent **yangi tilga
o'tib ketadi**:
- [arXiv:2609.01016](https://arxiv.org/pdf/2609.01016): zero-shot multilingual TTS modellari
  *"suffer from severe cross-lingual accent leakage resulting in unnatural foreign accent when
  the reference voice prompt language doesn't match the target text"*.
- [X-Voice, arXiv:2605.05611](https://arxiv.org/abs/2605.05611): faqat **matn darajasida** til
  shartlash yetarli emas — *"textual language conditioning alone still suffer[s] from accent
  leakage during cross-lingual synthesis"*. Ularning yechimi — **ikki darajali til injeksiyasi**
  (matn darajasi + vaqt darajasi).

**Nega mexanizm bo'yicha shunday?** Chunki (b) yo'lida model shunchaki **"oldingi tovushga
o'xshash bo'l"** qoidasiga amal qiladi. Aksent ham tovushda — model uni tembrdan ajratmaydi,
chunki ajratishga hech kim o'rgatmagan.

**XTTS tajribasi bu muammoni kam resursli tilda ko'rsatadi:** 16 til, lekin chex 52.4 soat,
yaponcha 57.3 soat. Maqola tan oladi: *"The different recording conditions are a challenge for
the generalization of the ZS-TTS models. Speakers who have a voice that differs greatly from those
seen in training also become a challenge."* Hamda SMOS'da XTTS raqobatchilardan
**−0.31 ... −0.39** past bo'ldi va buni *"complexity of massive multilingual training"* bilan
izohladi — ya'ni **ko'p tilni bir modelga tiqish spiker o'xshashligiga zarar yetkazadi**.

## 5.5. Referensdagi SHOVQIN ham klonlanadi

Bu (b) yo'lining eng og'riqli yon ta'siri, va VALL-E buni **yutuq** deb e'lon qilgan edi:
*"VALL-E could preserve the speaker's emotion and **acoustic environment** of the acoustic prompt
in synthesis."*

Lekin "akustik muhitni saqlash" = **shovqinni, exoni, mikrofon sifatini ham klonlash**.

Hujjatlashtirilgan topilmalar:
- [arXiv:2406.05699](https://arxiv.org/abs/2406.05699) ("An Investigation of Noise Robustness for
  Flow-Matching-Based Zero-Shot TTS"): *"the quality of the generated speech significantly
  deteriorates when the audio prompt contains noise"*, va *"limited research has been conducted
  to address this issue"*. Yechim sifatida: masked speech denoising bilan unsupervised pre-training,
  DNSMOS asosida filtrlash, **tasodifiy shovqin qo'shib fine-tuning** — bu uchtasi promptga
  speech enhancement qo'llashdan yaxshiroq natija berdi.
- [arXiv:2505.13830](https://arxiv.org/abs/2505.13830): *"LLM-based zero-shot TTS methods tend to
  preserve the acoustic environment of the audio prompt, leading to degradation in synthesized
  speech quality when the audio prompt contains noise."*

**Eng muhim va eng g'alati topilma — "enhancement paradoksi":** promptni shovqindan tozalasang,
audio sifat ballari yaxshilanadi, lekin **spiker o'xshashligi pasayadi**. Ya'ni enhancement
tembrning bir qismini ham olib ketadi. Bu savdolashuvdan qutulish yo'li yo'q.
> ⚠️ Bu detal men topgan qidiruv natijasi xulosasidan keldi (2406.05699 / 2505.13830 bo'yicha);
> StyleTTS-2 va HierSpeech++ uchun aniq raqamlarni asl maqolaning PDF'idan **tekshirmadim**.
> Ya'ni yo'nalish ishonchli, aniq raqamlar — **ma'lumot topilmadi**.

Qizig'i: shovqinli promptda **WER barqaror qolib, SIM pasayadi** — ya'ni model matnni to'g'ri
aytadi, lekin ovoz boshqa odamga o'xshaydi.

## 5.6. Prosodiya aynan ko'chirilib qolishi

Bu **ikki yo'lning ikkalasida ham** bor, lekin boshqa shaklda:

**Yo'l (a):** SV2TTS: *"The model is also not able to completely isolate the speaker voice from
the prosody of the reference audio"* — sintez ba'zan referensning prosodiyasini taqlid qiladi.
Ya'ni referens g'amgin bo'lsa, natija ham g'amgin chiqadi — xohlasang ham, xohlamasang ham.

**Yo'l (b):** bu yerda muammo kuchliroq, chunki model **butun akustik kontekstni** ko'chiradi.
Seed-TTS alohida nuqson aytadi: AR model *"shows consistency bias — maintaining uniform prosody
across long utterances rather than varying it naturally like human speech"*. Ya'ni uzun gapda
model boshidagi prosodiyani oxirigacha tortib ketadi — odam esa o'zgartiradi.

**Yechimlar:**
- OpenVoice: butun arxitekturani **tembr va uslubni ajratishga** qurdi (tone color converter
  flow tembrni olib tashlaydi, base TTS uslubni mustaqil beradi).
- NaturalSpeech 3 / FACodec: kontent, prosodiya, akustik tafsilot, tembrni **alohida subfazolarga**
  ajratdi.
- Seed-TTS: **self-distillation** — faqat bitta atributi farq qiladigan sun'iy juftliklar yasab,
  tembrni ajratishni o'rgatadi. Hamda **RL post-training** robustlik, spiker o'xshashligi va
  nazorat uchun. Lekin ogohlantiradi: faqat WER'ni optimallashtirganda **"reward hacking"**
  bo'ladi — tabiiylik qurbon bo'ladi.

## 5.7. Qisqa referens vs uzun referens

- **Yo'l (a):** SV2TTS **2 sekundda** deyarli maksimal natijaga chiqadi va **5 sekunddan keyin
  umuman yaxshilanmaydi**, chunki *"constrained by the small capacity of the speaker embedding"*.
  **Ya'ni 10 sekund berishning ma'nosi yo'q — 2 sekund ham shunday natija beradi.**
- **Yo'l (b):** VALL-E **3 sekund** prompt bilan ishlaydi. VALL-E 2'da Grouped Code Modeling
  *"particularly beneficial for longer prompts (10s+ on VCTK)"* — ya'ni uzun prompt foyda beradi,
  lekin ketma-ketlik uzunligi muammosi chiqadi.
- **Fine-tuning bilan solishtirish (YourTTS):** 20 sekund bilan fine-tune qilganda SECS
  0.743 → **0.930** ko'tarildi (zero-shot Sim-MOS 2.77 → 4.43). Lekin maqola ogohlantiradi:
  *"more than 45 seconds are more adequate to allow higher quality"* — 45 sekunddan kam bo'lsa
  o'xshashlik o'sadi, **lekin MOS (tabiiylik) pasayadi**.

> **Amaliy xulosa:** 10 sekund (b) yo'li uchun yetarli-yaxshi. (a) yo'li uchun esa 10 sekund
> **ortiqcha** — model uni ishlatolmaydi.

---

# 6-BOB. XULOSA — qaysi yo'l yolg'iz dasturchi uchun real?

## 6.1. Mexanizmning umumiy xaritasi

```
                        ZERO-SHOT TTS QUVURI
                        ════════════════════

  10 sek referens ──┬──> [(a) SPEAKER ENCODER] ──> 256-dim vektor ─┐
                    │     ECAPA / d-vector                          │
                    │     ↳ bottleneck, prosodiya yo'q              │
                    │                                               │
                    └──> [(b) CODEC / MEL] ──> to'liq token/mel ────┤
                          EnCodec / X-codec2 / mel                  │
                          ↳ bottleneck yo'q, hammasi o'tadi         │
                                                                    │
  matn ────────────────────────────────────────────────────────────┬┘
                                                                   │
                                                                   v
            ╔══════════════════════════════════════════════════════════╗
            ║              GENERATIV DVIGATEL                          ║
            ║  ┌──────────────┬────────────────┬──────────────────┐   ║
            ║  │ 1. AR codec  │ 2. Flow        │ 3. Masked        │   ║
            ║  │    LM        │    matching    │    generative    │   ║
            ║  │ (GPT uslubi) │ (ODE, 16-32    │ (iterativ        │   ║
            ║  │ 600+ o'tish  │  qadam)        │  ~27-60 o'tish)  │   ║
            ║  └──────────────┴────────────────┴──────────────────┘   ║
            ╚══════════════════════════════════════════════════════════╝
                                   │
                    ┌──────────────┴──────────────┐
                    v                              v
            diskret tokenlar                   mel-spektrogramma
                    │                              │
                    v                              v
          [CODEC DECODER]                  [VOCODER: Vocos/BigVGAN]
                    │                              │
                    └──────────────┬───────────────┘
                                   v
                                tovush
```

## 6.2. Yolg'iz dasturchi nimani realistik takrorlashi mumkin?

Tartiblangan ro'yxat (eng realdan eng realsizga):

| | Nima | Nega real / real emas |
|---|---|---|
| **1** | **Tayyor modelni ishlatish / fine-tune qilish** | Hamma zero-shot model allaqachon "ovozlar fazosi"ni o'rgangan. Senga faqat shu fazodan foydalanish kerak |
| **2** | **(b) yo'li + flow matching (E2/F5 uslubi), nolda o'qitish** | Eng kam qism: speaker encoder **yo'q**, duration model **yo'q**, G2P **yo'q**, codec **yo'q** (mel + tayyor Vocos). Bitta DiT + bitta loss. P-Flow 260 soatda ishlatgan |
| **3** | **(a) yo'li (SV2TTS uslubi)** | Texnik jihatdan oson (tayyor ECAPA olib, VITS'ga ulash), lekin natija **tubdan** cheklangan — SECS ~0.34–0.37 darajasida. CorentinJ o'zi repo'si haqida *"quickly gotten old"* deydi va Chatterbox'ga yo'naltiradi |
| **4** | **AR codec LM (VALL-E/Llasa uslubi) nolda** | Codec + AR + NAR + sampling trikları + beqarorlik + 600 token/sek. Hard sentences'da 68 % xato. Yolg'iz odam uchun eng og'riqli yo'l |
| **5** | **Masked generative (MaskGCT/SoundStorm uslubi)** | Ikki bosqichli quvur + ikkita tokenizer + mask jadvali. Eng ko'p harakatlanadigan qism |

### Nega aynan (2) — E2/F5 uslubi?

Chunki bu arxitektura **ataylab olib tashlash** orqali qurilgan. E2-TTS nomi ham shundan:
"Embarrassingly Easy". Yo'q qilingan narsalar ro'yxati:

```
  ✗ speaker encoder        — prompt o'zi yetadi
  ✗ duration model         — filler token + belgi nisbati
  ✗ phoneme aligner (MFA)  — alignment supervision kerak emas
  ✗ G2P / fonema lug'ati   — belgilar bilan ishlaydi (o'zbek uchun KATTA plus)
  ✗ text encoder           — F5'da faqat ConvNeXt bloklari
  ✗ neural codec           — mel yetadi, Vocos tayyor
  ✗ AR beqarorligi         — sampling hiylalariga hojat yo'q

  ✓ qoladi: bitta DiT + bitta flow matching loss + tayyor vocoder
```

Qolgan yagona haqiqiy to'siq — **ma'lumot**. Va bu to'siq **mexanizmdan kelib chiqadi**,
dasturlash murakkabligidan emas: model o'qimagan ovoz turini ekstrapolyatsiya qilolmaydi.
Shuning uchun yolg'iz dasturchi uchun eng mantiqiy yo'l — **tayyor ko'p-tilli zero-shot
modelni olib, o'zining maqsadli tili/ovozlari bo'yicha moslashtirish**, nolda o'qitish emas.
(Buning amaliy tafsilotlari — boshqa hujjatlarda.)

---

## 7. MANBALAR (hammasi tekshirilgan)

### Yo'l (a): speaker embedding
- **SV2TTS** — Jia va boshq., 2018. *Transfer Learning from Speaker Verification to Multispeaker TTS.* [arXiv:1806.04558](https://arxiv.org/abs/1806.04558)
- **GE2E** — Wan va boshq., 2017. *Generalized End-to-End Loss for Speaker Verification.* [arXiv:1710.10467](https://arxiv.org/abs/1710.10467)
- **ECAPA-TDNN** — Desplanques va boshq., 2020. [arXiv:2005.07143](https://arxiv.org/abs/2005.07143)
- **YourTTS** — Casanova va boshq., 2022. [arXiv:2112.02418](https://arxiv.org/abs/2112.02418)
- **OpenVoice** — Qin va boshq., 2023. [arXiv:2312.01479](https://arxiv.org/abs/2312.01479)
- **XTTS** — Casanova va boshq., 2024. [arXiv:2406.04904](https://arxiv.org/abs/2406.04904)
- **Real-Time-Voice-Cloning** (SV2TTS implementatsiyasi) — [github.com/CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

### Yo'l (b): in-context / prompt
- **VALL-E** — Wang va boshq., 2023. *Neural Codec Language Models are Zero-Shot TTS Synthesizers.* [arXiv:2301.02111](https://arxiv.org/abs/2301.02111)
- **VALL-E 2** — 2024. [arXiv:2406.05370](https://arxiv.org/abs/2406.05370)
- **P-Flow** — NVIDIA, NeurIPS 2023. [papers.nips.cc PDF](https://papers.nips.cc/paper_files/paper/2023/file/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Paper-Conference.pdf)
- **Voicebox** — Le va boshq., 2023. [arXiv:2306.15687](https://arxiv.org/abs/2306.15687)
- **E2-TTS** — 2024. [arXiv:2406.18009](https://arxiv.org/abs/2406.18009)
- **F5-TTS** — 2024. [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) · [github.com/SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- **Seed-TTS** — 2024. [arXiv:2406.02430](https://arxiv.org/abs/2406.02430)
- **CosyVoice 2** — 2024. [arXiv:2412.10117](https://arxiv.org/abs/2412.10117)
- **Fish-Speech** — 2024. [arXiv:2411.01156](https://arxiv.org/abs/2411.01156)
- **Llasa** — 2025. [arXiv:2502.04128](https://arxiv.org/abs/2502.04128)
- **Spark-TTS** — 2025. [arXiv:2503.01710](https://arxiv.org/abs/2503.01710)

### Neural audio codec
- **SoundStream** — Zeghidour va boshq., 2021. [arXiv:2107.03312](https://arxiv.org/abs/2107.03312)
- **EnCodec** — Défossez va boshq., 2022. [arXiv:2210.13438](https://arxiv.org/abs/2210.13438)
- **DAC** — Kumar va boshq., 2023. [arXiv:2306.06546](https://arxiv.org/abs/2306.06546)
- **SNAC** — [github.com/hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)
- **WavTokenizer** — 2024. [arXiv:2408.16532](https://arxiv.org/abs/2408.16532)
- **Mimi / Moshi** — Défossez va boshq., 2024. [arXiv:2410.00037](https://arxiv.org/abs/2410.00037)

### Masked generative
- **SoundStorm** — Borsos va boshq., 2023. [arXiv:2305.09636](https://arxiv.org/abs/2305.09636)
- **MaskGCT** — 2024. [arXiv:2409.00750](https://arxiv.org/abs/2409.00750)
- **NaturalSpeech 3** — 2024. [arXiv:2403.03100](https://arxiv.org/abs/2403.03100)

### Vocoder
- **BigVGAN** — 2022. [arXiv:2206.04658](https://arxiv.org/abs/2206.04658)
- **Vocos** — 2023. [arXiv:2306.00814](https://arxiv.org/abs/2306.00814)

### Ma'lumot va nuqsonlar
- **Emilia** — 2024. [arXiv:2407.05361](https://arxiv.org/abs/2407.05361)
- **RALL-E** — 2024. [arXiv:2404.03204](https://arxiv.org/abs/2404.03204)
- **Noise robustness (flow matching)** — 2024. [arXiv:2406.05699](https://arxiv.org/abs/2406.05699)
- **Noise robustness (LLM-based)** — 2025. [arXiv:2505.13830](https://arxiv.org/abs/2505.13830)
- **Voice Cloning: Comprehensive Survey** — Azzuni & El Saddik, 2025. [arXiv:2505.00579](https://arxiv.org/abs/2505.00579)
- **Accent leakage (code-switching)** — [arXiv:2609.01016](https://arxiv.org/pdf/2609.01016)
- **X-Voice (accent leakage)** — [arXiv:2605.05611](https://arxiv.org/abs/2605.05611)
- **Data diversity in LLM-TTS fine-tuning** — Purwar & Choudhary, 2026. [arXiv:2603.10904](https://arxiv.org/html/2603.10904) ⚠️ kichik preprint

---

## 8. MA'LUMOT TOPILMADI — ochiq qolgan savollar

Halol bo'lish uchun, quyidagilarni **topa olmadim**:

1. **Jami soatni qotirib faqat spiker sonini o'zgartirgan nazoratli ablatsiya**
   zero-shot TTS **sintezatori** uchun — ma'lumot topilmadi. (SV2TTS'da bor, lekin u
   speaker **encoder** uchun, sintezator uchun emas.)
2. **Emilia datasetidagi spikerlar soni** — maqolada keltirilmagan (101 654 soat, 6 til bor,
   spiker soni yo'q).
3. **Shovqinli promptda StyleTTS-2 / HierSpeech++ uchun aniq SIM/DNSMOS raqamlari** —
   qidiruv xulosasida yo'nalish bor, asl PDF'dan tekshirilmagan.
4. **F5/E2'ning belgi-nisbati duration taxminining o'zbek lotin yozuvida qanday ishlashi** —
   hech bir manbada ko'rilmagan; yuqorida bu mexanizmdan kelib chiqqan taxmin deb belgilangan.
5. **DAC'ning TTS sifatiga ta'siri haqida to'g'ridan-to'g'ri qiyoslash** (EnCodec vs DAC
   bir xil TTS modelida) — ma'lumot topilmadi.

### Qidiruv budjeti to'g'risida
WebSearch limiti **oshib ketmadi**: ruxsat etilgan 18 ta chaqiruvdan **5 tasi** ishlatildi;
qolgan barcha ma'lumot arXiv HTML / ar5iv sahifalari, GitHub README'lari va NeurIPS PDF'idan
to'g'ridan-to'g'ri olindi.

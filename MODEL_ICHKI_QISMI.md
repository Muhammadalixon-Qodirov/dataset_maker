# Gapiruvchi model ichidan qanday ishlaydi

**Savol:** TTS modellarining ichki qismi qanday ishlaydi? Unga "ovozimga moslashib gapirishi" uchun nimadir qo'shsa bo'ladimi, va u qism ichdan qanday ishlaydi?

**Sana:** 2026-10-01
**Manba:** `tadqiqot/zeroshot/arxitektura.md` (1384 qator, ~40 tekshirilgan manba)

---

## 0. Qisqa javob

1. Har qanday TTS model **3 qismdan** iborat: frontend (matn→token) → akustik model (token→mel) → vocoder (mel→tovush).
2. Ovoz identitetini kiritishning **aniq 3 joyi** bor: vektor sifatida (embedding), prefiks sifatida (prompt), yoki chiqishdan keyin o'girish (RVC).
3. "Qo'shish" texnik jihatdan **LoRA** deb ataladi va u aynan qo'shish bilan ishlaydi — katta vaznlar muzlatiladi, yoniga ikki ingichka matritsa qo'yiladi.
4. Shuning uchun **LoRA zero-shot'ni saqlaydi, to'liq fine-tune esa o'ldiradi.**
5. LoRA'ning peak VRAM = **6.96 GB** → RTX 3050 (8 GB) ga sig'adi.

---

## 1. Uch qism

```
  MATN                                                       TOVUSH
   │                                                           ▲
   ▼                                                           │
┌───────────┐       ┌──────────────────┐       ┌──────────────────┐
│ 1.FRONTEND│ ────▶ │ 2.AKUSTIK MODEL  │ ────▶ │ 3.VOCODER        │
│  matn →   │tokens │   "yurak"        │  mel  │  mel → to'lqin   │
│  tokenlar │       │  (DiT yoki GPT)  │       │ (Vocos,BigVGAN)  │
└───────────┘       └──────────────────┘       └──────────────────┘
```

### 1.1 Frontend — matnni raqamga aylantirish

```
"Assalomu alaykum"  →  [12, 45, 45, 3, 9, 15, ...]
```

| Usul | Nima qiladi | Kim ishlatadi | O'zbek uchun |
|---|---|---|---|
| **Belgi (char)** | Har harf = 1 token | F5-TTS, Chatterbox | 🟢 To'g'ridan-to'g'ri ishlaydi |
| **Subword (BPE)** | So'z bo'laklari | OmniVoice (Qwen3), XTTS | 🟢 Ishlaydi (vocab kengaytirish kerak bo'lishi mumkin) |
| **Fonema (G2P)** | `sh` → `ʃ` | Piper, eski modellar | 🔴 **Yaroqsiz** — ishonchli o'zbek G2P yo'q |

**Nega fonema usuli o'zbek uchun yopiq:**
- `CharsiuG2P uzb` — mualliflarning o'zi *"do not use it"* deydi
- `espeak-ng uz` — literal **3 qatorli fayl**, `status testing`, ~90 qoida, ona tilida so'zlashuvchi tekshirmagan

👉 Shuning uchun **F5-TTS va OmniVoice qulay** — ularga xom lotin matni kiradi.

### 1.2 Akustik model — "yurak"

Tokenlarni **tovush tasviriga** aylantiradi. Hali tovush emas — tovushning *rasmi*: **mel-spektrogramma**.

```
Chastota
  ▲
  │ ███░░░███░░░░███      ← har ustun = 1 kadr (~10 ms)
  │ ██░░░░██░░░░░██       ← har ustunda ~80–100 son
  │ █░░░░░░█░░░░░░█
  └──────────────────▶ Vaqt
```

**Nega to'g'ridan-to'g'ri to'lqin yasalmaydi** — raqamlar aytib beradi:

```
1 sekund tovush, 24 kHz      =  24,000 ta son       ← juda ko'p
1 sekund mel-spektrogramma   =  50 kadr × 80 son    ← ~6× kam
1 sekund audio token         =  12.5–50 token       ← ~1000× kam
```

Model sekundda 24,000 sonni taxmin qila olmaydi; 50 tokenni — osongina. **Siqish butun ishni mumkin qiladi.**

### 1.3 Vocoder

Mel-spektrogrammani haqiqiy to'lqinga o'giradi. Bu **hal qilingan masala** — tayyor `Vocos` yoki `BigVGAN` olinadi, o'qitish kerak emas.

---

## 2. Neural audio codec — "audio token" nima

Flow matching modellari to'g'ridan-to'g'ri **mel** bilan ishlaydi (uzluksiz). AR modellar esa **diskret token** talab qiladi — chunki GPT mashinasi faqat diskret token ustida ishlaydi.

### Nega RVQ (Residual Vector Quantization) kerak

Bitta kodkitob bilan 80 bit/kadr kodlash uchun **2^80 vektor** kerak bo'lardi — imkonsiz. Yechim: kaskad. Birinchi kodkitob qo'pol taxmin qiladi, ikkinchisi **qolgan xatoni** kodlaydi, uchinchisi uning xatosini, va hokazo.

### Frame rate — hayot-mamot masalasi

| Codec | Tezlik |
|---|---|
| EnCodec | 600 token/sek |
| CosyVoice 2 | 25 Hz |
| Mimi (Moshi) | **12.5 Hz** |

Frame rate past bo'lsa — model uzun gapni eslab qoladi va tez ishlaydi. Yuqori bo'lsa — sifat yaxshi, lekin sekin va xotira yetmaydi.

---

## 3. 🎯 Ovoz qayerdan kiradi — aniq 3 joy

```
          ┌─────────────────────────────────────┐
MATN ───▶ │        AKUSTIK MODEL                │ ──▶ mel ──▶ TOVUSH
          │                                     │              │
          │   ▲(1)          ▲(2)                │              │(3)
          └───┼─────────────┼───────────────────┘              │
              │             │                                  ▼
        [256 son]      [10 sek audio]                    [RVC o'giradi]
        embedding        prefiks
```

| # | Joy | Usul | Kim ishlatadi |
|---|---|---|---|
| **1** | Modelga **vektor** | Speaker embedding | SV2TTS (2018), YourTTS |
| **2** | Modelga **prefiks** | Davom ettirish (in-context) | VALL-E, F5-TTS, OmniVoice |
| **3** | **Chiqishdan keyin** | Voice conversion | **RVC** |

### Nega (2) eng yaxshi — o'lchangan dalil

P-Flow (NeurIPS 2023): **bir xil arxitektura, bir xil 260 soat**, faqat conditioning almashtirilgan:

| Usul | SECS (o'xshashlik) |
|---|---|
| (1) embedding | 0.373 |
| **(2) prompt** | **0.544** |

### (1) nega cheklangan

- **Bottleneck:** SV2TTS maqolasining o'zi aytadi — 5 sekunddan uzun referens **yordam bermaydi**. 256 songa ko'proq sig'maydi.
- **GE2E loss** bir odamning barcha gaplarini bitta markazga tortadi → **prosodiyani ataylab o'chiradi**.

> *Politsiya suratchisi:* ovozni 256 so'z bilan tasvirlab, rassomga chizdirasiz. Tanib olinadi — lekin aynan o'sha emas.

### (2) nega ishlaydi — konseptual sakrash

Model faqat bitta narsaga o'qitilgan:

```
matn + shu paytgacha bo'lgan tovush  →  keyingi tovush
```

Haqiqiy yozuvlarda **gap o'rtasida odam o'zgarmaydi.** Shuning uchun xatoni kamaytiradigan yagona strategiya — *"oldingi tovush bilan bir xil bo'l"*.

Inference'da prefiks o'rniga **boshqa odamning** yozuvi qo'yiladi. Model o'z qoidasiga sodiq qolib o'sha ovozda davom etadi.

> ### Klonlash — maxsus funksiya EMAS. Bu izchillik qoidasining yon ta'siri.
> Modelda "speaker" tushunchasi **umuman yo'q**.

> *Kserokopiya:* original shishada jismonan yotadi. Hech narsa tasvirlanmaydi, siqilmaydi.

### (3) RVC — qayerda turadi

```
TTS  :  MATN  →  ovoz
RVC  :  OVOZ  →  boshqa ovoz     ← matn kerak emas
```

Bu "ustiga yopishtirilgan" yechim — arzon, matn kerak emas, lekin sifat cheklangan, chunki ovoz modelning **ichida** tug'ilmaydi, keyin o'giriladi.

---

## 4. F5-TTS ning "bo'shliqni to'ldirish" hiylasi

Bu tushunishga arziydi: aynan shu tufayli **speaker encoder ham, duration model ham kerak emas.**

Vazifa qayta ta'riflanadi — "matndan nutq yasash" emas, **"nutqning o'chirilgan qismini tiklash"**.

### O'qitishda

```
Audio:  [███████]  [░░░░ O'CHIRILGAN ░░░░]  [███████]
Matn:   "to'liq matn — hammasi beriladi"
Vazifa: o'chirilgan joyni tikla
```

### Ishlatishda

```
Beramiz:  [SIZNING 10 SEK]  [░░░ BO'SH ░░░]
Matn:     "referens matn"  +  "yangi matn"
              ↓
Model bo'shliqni to'ldiradi — atrofga mos kelishi uchun
SIZNING ovozingizda to'ldiradi
```

### Duration qanday hal bo'ladi

Matn **filler token** bilan to'liq uzunlikka cho'ziladi; model har so'zga qancha joy berishni o'zi hal qiladi. F5-TTS izohi: *"the model learns to assign the length of each word or phoneme according to the given total sequence length, resulting in improved prosody and rhythm."*

Umumiy uzunlik oddiy nisbat bilan:
```
len(yangi matn) / len(referens matn) × len(referens audio)
```

### ⚠️ Keng tarqalgan xato

"Flow matching = duration model yo'q" degan gap **noto'g'ri**. **Voicebox** ham flow matching, lekin unda duration model **bor** (phone-darajali, default regressiya). Duration'ni butunlay yo'q qilgan — faqat **E2-TTS va F5-TTS**.

### Sway Sampling (F5-TTS)

Flow qadamlarini bir tekis emas, boshiga to'playdi:
```
f_sway(u, s) = u + s · ( cos(π/2 · u) − 1 + u ),    s ∈ [−1, 2/(π−2)]
```

---

## 5. Sintez jarayoni — ichida nima sodir bo'ladi

Flow matching uslubida:

```
1-qadam:   [████ SHOVQIN ████]              ← tasodifiy shovqindan
              ↓  (matn + referens "ramka" bo'lib cheklaydi)
8-qadam:   [▓▓ nutqqa o'xshay boshladi ▓▓]
              ↓
16-qadam:  [░ deyarli tayyor ░]
              ↓
32-qadam:  [ MEL-SPEKTROGRAMMA ]  →  Vocos  →  TOVUSH
```

> *Haykaltarosh:* marmar bo'lagidan boshlaydi, har o'tishda aniqroq qiladi.

Jami **16–32 qadam**. AR modellar (VALL-E) esa **600–6000 qadam** — har token uchun bittadan.

---

## 6. Uch avlod — haqiqiy raqamlar

| | AR codec LM | **Flow matching** | Masked generative |
|---|---|---|---|
| Modellar | VALL-E, XTTS, Fish-Speech, Llasa | **E2-TTS, F5-TTS** | MaskGCT, SoundStorm, NS3, OmniVoice |
| Qadamlar (10 s) | 600–6000 | **16–32 NFE** | 27 (SoundStorm) – 60 (NS3) |
| Tezlik | 10 s audio → ~6.2 s | **RTF 0.15** (16 NFE), Voicebox 20× tez | ~100× AudioLM'dan |
| Duration | Model o'zi (`<eos>`) | Voicebox: **kerak**; E2/F5: **kerak emas** | Kerak emas (mask orqali) |
| Barqarorlik | 🔴 **Eng yomon** — VALL-E hard sentences'da **68% xato** | 🟢 WER 2.42 (F5), 1.9 (E2) | 🟢 WER 2.63 (MaskGCT) |
| Streaming | 🟢 **Oson** — Fish-Speech first-packet 150 ms | 🔴 Qiyin (CosyVoice2 chunk-aware bilan yechdi) | 🔴 Qiyin |
| Prosodiya | Ekspressiv, lekin beqaror | Barqaror, ba'zan tekisroq | Barqaror |
| LLM checkpointdan | 🟢 **Ha** — Llasa (LLaMA), Spark-TTS (Qwen2.5) | ❌ | ❌ |
| Codec kerakmi | **Ha, majburiy** | **Yo'q** — ko'pincha to'g'ridan mel | Ha |
| Eng yaxshi SIM-o | VALL-E 2: 0.643 | **E2-TTS: 0.708** | MaskGCT: 0.687 |

🔥 **E2-TTS SIM-o = 0.708, haqiqiy odam yozuvi = 0.695.** O'xshashlik bo'yicha ground truth'dan o'tgan.

⚠️ Halol: E2-TTS testning **7%** ida alignment buziladi.

---

## 7. 🎯 "Shunday narsa qo'shsa bo'ladimi?" — javob

**Ha.** To'rt darajasi bor:

| Daraja | Nima o'zgaradi | Ma'lumot | Zero-shot saqlanadimi | VRAM |
|---|---|---|---|---|
| **1. Zero-shot** | **Hech narsa** — faqat ishora | 10 sekund | ✅ hech narsa buzilmaydi | 2–4 GB |
| **2. LoRA** | **Yoniga kichik qatlam qo'shiladi** | 30 daq – 10 soat | ✅ **saqlanadi** | **6.96 GB** |
| **3. To'liq fine-tune** | Barcha vaznlar siljiydi | 10+ soat | ❌ **o'ladi** | 12 GB+ |
| **4. Noldan** | Hammasi yangi | 20–25+ soat | — | ko'p GPU |

### LoRA ichdan qanday ishlaydi

```
ODDIY FINE-TUNE                    LoRA
───────────────                    ────
┌─────────────┐                    ┌─────────────┐   ┌───┐
│  KATTA      │ ← hammasi          │  KATTA      │ + │ A │ ← faqat
│  VAZN       │   o'zgaradi        │  VAZN       │   │ B │   bular
│  MATRITSASI │                    │ (MUZLATILGAN)│  └───┘   o'rganadi
└─────────────┘                    └─────────────┘
  baza bilim: BUZILADI               baza bilim: SAQLANADI
```

Katta matritsaga **tegilmaydi** — muzlatiladi. Yoniga ikki ingichka matritsa qo'shiladi (rank `r`, masalan r=32), faqat ular o'rganadi.

### 💡 Shuning uchun jumboq yechiladi

Tadqiqotda ikki zid natija chiqqan edi:

```
"Bitta spikerda fine-tune → zero-shot O'LADI"       (F5-TTS jamoasi, #143)
"30 soat bitta spiker LoRA → zero-shot SAQLANDI"    (Abduqayum, o'zbek Chatterbox)
```

Sabab:

| Usul | Nima bo'ladi |
|---|---|
| To'liq fine-tune | Vaznlar siljiydi → **"ovozlar fazosi" ustiga yoziladi** → model faqat bitta ovozni biladi |
| **LoRA** | Baza muzlatilgan → **"ovozlar fazosi" joyida qoladi** → ustiga sizning ovozingiz qo'shiladi |

👉 **Reja ishlaydi — faqat LoRA bilan.** Va 6.96 GB RTX 3050 ga sig'adi.

⚠️ Majburiy nazorat: fine-tune'dan oldin/keyin **o'zbek bo'lmagan (OOD) spikerlarda SIM-o** o'lchang. Tushsa — catastrophic forgetting boshlangan, to'xtatish kerak.

---

## 8. Nega ulkan ko'p-spikerli dataset kerak

Model bitta ovozni emas, **ovozlar fazosini** o'rganadi.

| Tajriba | Natija |
|---|---|
| NaturalSpeech 3: 1K → 200K soat | Sim-O 0.64 → 0.73 |
| Llasa: **ma'lumot** 3× | SIM **+0.031** |
| Llasa: **model** 8× | SIM **+0.002** ← deyarli nol |

> **Soatlar talaffuzni sotib oladi. Spikerlar xilma-xilligi o'xshashlikni sotib oladi.**
> Katta GPU emas — katta dataset kerak.

**Zid dalil:** P-Flow **260 soatda** ishlagan (95,000 emas). Ya'ni kichikroq zero-shot model printsipial mumkin.

---

## 9. Ma'lum nuqsonlar

| Nuqson | Qayerda | Dalil |
|---|---|---|
| Hallucination, takrorlanish, so'z tushishi | AR modellar | VALL-E hard sentences'da **68% xato** |
| Alignment buzilishi | NAR ham immun emas | E2-TTS testning **7%** ida |
| Duration beqarorligi | AR | `<eos>` erta yoki kech |
| Referens shovqini klonlanadi | Hammasida | Toza referens majburiy |
| Prosodiya aynan ko'chiriladi | Prompt usuli | Referens ohangi natijaga o'tadi |
| Aksent oqib o'tishi | Cross-lingual | Referens tili natijaga ta'sir qiladi |

---

## 10. Sizning holatingiz uchun xulosa

| Qarorlar | Tanlov | Sabab |
|---|---|---|
| Frontend | **Belgi yoki subword** | O'zbek G2P yaroqsiz |
| Ovoz kiritish | **(2) prompt** | SECS 0.544 vs 0.373 |
| Avlod | **Flow matching** | Barqaror (AR'da 68% xato) |
| Moslashtirish | **LoRA, to'liq FT emas** | Zero-shot saqlanadi + 8 GB ga sig'adi |
| Vocoder | **Tayyor Vocos** | Hal qilingan masala |
| Baza model | OmniVoice (NC) yoki Chatterbox (MIT) | Litsenziyaga qarab |

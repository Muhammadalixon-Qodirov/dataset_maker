# Ovoz klonlash (zero-shot) — tadqiqot xulosasi

**Savol:** modelga 10 sekund ovoz berilsa, shu ovozda gapira boshlaydigan tizim qurish mumkinmi? Kimlar qilgan, qanday qilgan?

**Tadqiqot sanasi:** 2026-10-01
**Batafsil:** `tadqiqot/zeroshot/` — 5 fayl (arxitektura, kim_qilgan, arzon_yol, ozbek_yoli, 8gb_reallik)

---

## 0. Eng muhim xulosa — kutilmagan

**O'zbek tili uchun bu allaqachon mavjud va ishlaydi.**

**OmniVoice** (k2-fsa / Xiaomi, 2026) — 646 tilni qo'llaydigan zero-shot ovoz klonlash modeli. O'zbek tili uning **rasmiy ro'yxatida**, 115.28 soat o'qitish ma'lumoti bilan. Va sifati o'lchangan:

| FLEURS o'zbek testida | CER |
|---|---|
| Haqiqiy inson yozuvlari | 3.91% |
| **OmniVoice sinteziyasi** | **3.37%** |

Ya'ni model sintez qilgan o'zbek nutqi, ASR o'lchovi bo'yicha, **FLEURS'dagi haqiqiy odam yozuvlaridan tushunarliroq.**

👉 **Birinchi qadam — 25 kun ovoz yozish emas, 1 kun ichida OmniVoice'ni sinab ko'rish.**

---

## 1. Mexanizm — qanday ishlaydi

Zero-shot klonlash **ovozni o'rganish emas.** Model minglab odam ovozida gapirishni allaqachon o'rgangan; 10 sekundlik namuna faqat **ishora**.

### Ikki yo'l

**Yo'l (a) — speaker embedding** (SV2TTS, 2018)
Alohida tarmoq ovozni **256 songa** siqadi, keyin bu vektor sintezatorga beriladi.

```
[10 sek ovoz] → encoder → [256 son] → sintezator → nutq
```

> *Politsiya suratchisi:* ovozni 256 so'z bilan tasvirlab, rassomga chizdirasiz. Tanib olinadi — lekin aynan o'sha emas.

Nega cheklangan:
- **Bottleneck** — SV2TTS maqolasining o'zi aytadi: 5 sekunddan uzun referens yordam bermaydi
- **GE2E loss** bir odamning barcha gaplarini bitta markazga tortadi → **prosodiyani ataylab o'chiradi**

**Yo'l (b) — in-context davom ettirish** (VALL-E, 2023) ← konseptual sakrash

Model faqat bitta narsaga o'qitilgan:
```
matn + shu paytgacha bo'lgan tovush  →  keyingi tovush
```

Haqiqiy yozuvlarda gap o'rtasida odam o'zgarmaydi. Shuning uchun xatoni kamaytiradigan yagona strategiya — **"oldingi tovush bilan bir xil bo'l"**.

Inference'da prefiks o'rniga **boshqa odamning** yozuvini qo'yamiz. Model o'z qoidasiga sodiq qolib o'sha ovozda davom etadi.

> ### Klonlash — maxsus funksiya EMAS. Bu izchillik qoidasining yon ta'siri.
> Modelda "speaker" tushunchasi umuman yo'q.

> *Kserokopiya:* original shishada jismonan yotadi. Hech narsa tasvirlanmaydi.

### Eng toza dalil (P-Flow, NeurIPS 2023)

Bir xil arxitektura, bir xil 260 soat, faqat conditioning almashtirilgan:

| Usul | SECS (ovoz o'xshashligi) |
|---|---|
| (a) embedding | 0.373 |
| **(b) prompt** | **0.544** |

### Uch avlod backbone

| Avlod | Modellar | Afzalligi | Kamchiligi |
|---|---|---|---|
| AR codec LM | VALL-E, XTTS, Fish-Speech | Uzunlikni o'zi hal qiladi, streaming oson | Qiyin gaplarda **68% xato** — so'z tushadi/takrorlanadi |
| **Flow matching** | **E2-TTS, F5-TTS** | 16–32 qadam, **20× tez**, barqaror | — |
| Masked generative | MaskGCT, SoundStorm, OmniVoice | 30 sek — 27 o'tishda | — |

### Neural audio codec — (b) ni mumkin qilgan narsa

Tovushni **diskret tokenga** aylantirib, unga GPT mashinasini qo'llash mumkin bo'ladi. RVQ kaskad kerak, chunki 80 bit/freym uchun bitta kodkitobda 2^80 vektor kerak bo'lardi.

Frame rate hayot-mamot: EnCodec 600 tok/sek · Mimi 12.5 Hz · CosyVoice2 25 Hz.

### Nega ulkan dataset kerak

Model bitta ovozni emas, **ovozlar fazosini** o'rganadi.

| Tajriba | Natija |
|---|---|
| NaturalSpeech 3: 1K → 200K soat | Sim-O 0.64 → 0.73 |
| Llasa: **ma'lumot** 3× | SIM **+0.031** |
| Llasa: **model** 8× | SIM **+0.002** ← deyarli nol |

> **Soatlar talaffuzni sotib oladi. Spikerlar xilma-xilligi o'xshashlikni sotib oladi.**
> Katta GPU emas — katta dataset kerak.

---

## 2. Kim qurgan — tarix va holat

Jami **48 tizim** tekshirilgan.

```
2023 yanvar   VALL-E (Microsoft)   →  zero-shot klonlashni ISHLATTIRDI
                                      codec token + LM + 60K soat + 3 sek prompt
2024 oktyabr  F5-TTS               →  ARZON va takrorlanadigan qildi
                                      335.8M param, 8×A100, 1 hafta, ochiq ma'lumot
2025 may      Chatterbox (MIT)     →  YURIDIK JIHATDAN BEPUL qildi
                                      0.5M soat
2026          OmniVoice            →  646 TILGA yetkazdi (o'zbek ham)
```

### 🔴 Litsenziya tuzoqlari — eng ko'p xato qilinadigan joy

| Model | Haqiqiy holat |
|---|---|
| **F5-TTS** | Kod MIT, **vaznlar CC-BY-NC** (Emilia sababli). "F5-TTS MIT" degan gap — **xato**. Fine-tune natijasi ham NC bo'lib qoladi |
| **OmniVoice** | Kod Apache-2.0, **vaznlar CC-BY-NC** (yana Emilia) |
| **XTTS v2** | Coqui Public Model License — **tijorat TAQIQ**. `coqui.ai/cpml` o'lgan (404) |
| **MegaTTS3** | Nominal Apache-2.0, lekin **WaveVAE encoder vaznlari berilmagan** → ixtiyoriy klonlash aslida ishlamaydi |
| **VibeVoice** | MIT, lekin "research only", Microsoft **TTS kodini olib tashlagan** |
| **Higgs Audio v2** | `license: other` — Boson Community License (100k foydalanuvchigacha OK). v3 nokommersiyaga o'tdi |
| NC ro'yxati | MaskGCT, Vevo, Llasa, Spark-TTS, Fish-Speech 1.5, OpenAudio |

⚠️ **Tendensiya:** 2026 da litsenziyalar qattiqlashyapti (Higgs v2→v3, Fish→maxsus). Ochiq oyna yopilib boryapti.

### ✅ Haqiqatan tijoratga ruxsat beruvchilar

| Model | Litsenziya | Tillar | Hajm |
|---|---|---|---|
| **CosyVoice 2/3** | Apache-2.0 | 9 til + 18 dialekt | 0.5–1.5B |
| **Chatterbox** | MIT | 23+ til | 0.5B (Turbo 350M, Nano 110M) |
| **GPT-SoVITS** | MIT | — | — |
| Zonos, Sesame CSM-1B, Orpheus, Dia, OpenVoice | Apache/MIT | — | — |

⚠️ Chatterbox'da **majburiy Perth watermark**.

### Ochiq vs yopiq — halol taqqoslash

Seed-TTS (ByteDance) xitoy tilida insondan o'tgan:
```
Seed-TTS :  WER 1.115  |  SIM 0.796
Inson    :  WER 1.254  |  SIM 0.750
```

Lekin subyektiv baholashda (Artificial Analysis Elo):
```
ElevenLabs v4  :  1320   ← yopiq, yetakchi
Breeze TTS 2   :  1206   ← eng yaxshi ochiq
                  ~114 Elo farq
```

👉 Obyektiv o'lchovlarda deyarli teng, **odam qulog'iga sezilarli farq bor.**

**Aniqlik:** Kokoro, Parler-TTS, Maya1 — zero-shot klonlash **qilmaydi** (faqat tayyor ovozlar). Ko'pincha chalkashtiriladi.

---

## 3. O'zbek tili uchun

### OmniVoice — yagona o'lchangan variant

| Xususiyat | Qiymat |
|---|---|
| To'liq nomi | *OmniVoice: Towards Omnilingual Zero-Shot TTS with Diffusion Language Models* |
| Tashkilot | k2-fsa / Xiaomi (Daniel Povey senior muallif) |
| Maqola | arXiv:2604.00688 |
| Hajm | 0.6B parametr · 646 til · 581k soat |
| **O'zbek** | `uz`/`uzb` — **115.28 soat**; `uzn` (Northern Uzbek) — 15.23 soat |
| Turk | `tr`/`tur` — 125.36 soat |
| **FLEURS uz CER** | **3.37%** (ground truth 3.91%) |
| Mashhurligi | **1,431,614 yuklab olish**, 1,456 like |
| Litsenziya | Kod Apache-2.0 · **vaznlar CC-BY-NC** |
| LoRA | ✅ Rasmiy qo'llanadi — `run_finetune_lora.sh` |

### 🎯 Nega bu eng oson baza

OmniVoice **Qwen3-0.6B subword tokenizer** ishlatadi va maqolada aniq yozilgan: *"eliminat[es] cumbersome grapheme-to-phoneme conversion and language-specific text normalization."*

Ya'ni:
- **O'zbek lotini xom holda kiradi** — G2P kerak emas
- **Tokenizer kengaytirish kerak emas** → pretrained embedding buzilmaydi

Bu aynan XTTS-v2 (BPE, vocab kengaytirish kerak, 48–100 soat) va Chatterbox (vocab shape mismatch) ni og'ir qiladigan to'siq.

Va bu avvalgi tadqiqotdagi muammoni hal qiladi: o'zbek G2P resurslari zaif (`CharsiuG2P uzb` noto'g'ri, `espeak-ng uz` "testing"). G2P kerak bo'lmasa — muammo yo'q.

### 🎯 Ikkinchi muhim topilma — bitta spiker savoliga javob

Avval topilgan ogohlantirish: *"bitta spikerda fine-tune zero-shot'ni o'ldiradi."* Lekin o'zbek tilida aynan bu sinalgan:

**`Abduqayum/uzbek-tts-natural-speech-chatterbox`**
- Chatterbox'ning **LoRA** finetune'i
- **~30 soat bitta spikerli o'zbek audiokitob** (FeruzaSpeech'ga juda o'xshash sharoit)
- Tokenizer o'zgartirilmagan — *"2454-token grapheme tokenizer covers Uzbek's Latin script without modification"*
- **MIT litsenziya**
- Natija: **"zero-shot voice cloning remains functional. Users can pass any reference wav and it will speak your text in that voice"**

### 💡 Ziddiyatning yechimi

| Usul | Bitta spikerda | Natija |
|---|---|---|
| **To'liq SFT** | ovozlar fazosi qayta yoziladi | ❌ zero-shot o'ladi |
| **LoRA** | baza model saqlanadi, ustiga qo'shiladi | ✅ **zero-shot saqlanadi** |

👉 **LoRA — bitta spikerli ma'lumot bilan zero-shot'ni saqlashning yo'li.**
⚠️ Lekin bu modelda sifat metrikasi **nashr etilmagan**, faqat demo audio.

### ❌ Uchta o'lgan yo'l

| Yo'l | Nega o'lgan |
|---|---|
| Turkiy translit transfer | O'lchangan va **muvaffaqiyatsiz**: TurkicTTS o'zbek Q=2.85, **tushunarlilik 45%** |
| espeak-ng `uz` | Literal **3 qatorli fayl**, `status testing`, ~90 qoida, ona tilida so'zlashuvchi tekshirmagan → Zonos'ning espeak yo'li yopiq |
| FeruzaSpeech ni **baza** qilish | Faqat **sifat kalibratori** bo'lishi kerak, baza emas |

### Boshqa mavjud o'zbek modellari

| Model | Litsenziya | Zero-shot? | Holat |
|---|---|---|---|
| `Abduqayum/uzbek-tts-...-chatterbox` | **MIT** ✅ | ✅ **Ha** | Metrika yo'q, demo audio bor. **O'zbek uchun eng muhim ochiq urinish** |
| `uzlm/sayro-tts-1.7B` (UzLM, Examy.me) | `sayro-terms-of-use` — maxsus, qo'lda tasdiq | ❌ Yo'q (tayyor spikerlar) | Metrika yo'q. *"synthetic audio + public datasets"*, soat ko'rsatilmagan |
| `facebook/mms-tts-uzb-script_cyrillic` | CC-BY-NC | ❌ | **Faqat kirill** — lotin varianti HF'da yo'q (API bilan tasdiqlandi) |
| TurkicTTS (IS2AI) | akademik+tijorat | ❌ | O'zbek: Q=2.85, I=45% — past |

---

## 4. 8 GB VRAM realligi (RTX 3050)

### 🟢 Ishlatish — muammosiz

| Model | VRAM | 3050 da |
|---|---|---|
| F5-TTS | ~2–4 GB | ✅ 320 ms birinchi bo'lak |
| Chatterbox | ~2–3 GB | ✅ 280 ms |
| GPT-SoVITS | ≥6 GB (rasmiy) | ✅ |
| XTTS-v2 | ~2–6 GB | ✅ |

Sig'maydiganlar: Orpheus 3B (6.4 GB, OOM xavfi), Higgs Audio v2 (*"8 GB cards aren't enough"*), VibeVoice-Large (~20 GB).

### 🔴 Fine-tune — yetmaydi

```
F5-TTS eng past hujjatlashtirilgan config:  12 GB (4080 Laptop)
Sizda:                                       8 GB
```

⚠️ **Agentlar ziddiyati:** biri "3050 chegarada ishlaydi" (RTX 3080 10GB asosida), ikkinchisi "eng past 12 GB". Bittasi 4070 Laptop 8GB da sinagan — natija *"a wet sound, gargling like"*, lekin uning ma'lumoti faqat 1 soat 15 daqiqa edi. Ayb VRAM'da yoki ma'lumotda — ajratib bo'lmaydi.

**Yagona ishonchli mahalliy variant:** `GPT-SoVITS` — rasmiy hujjat *"batch_size 1 if your gpu has 6G vram"*. Windows one-click bor (`.7z` + `go-webui.bat`).

### Windows ikkinchi zarba

| Vosita | Holat |
|---|---|
| `bitsandbytes` (8-bit optimizer) | *"Windows support exists only for inference... for training, it's not viable"* ❌ |
| `deepspeed` | `LNK1181: cannot open input file 'aio.lib'` ❌ |
| RTX 3050 = **Ampere** | FP8/FP4 native yo'q — sig'dirish hiylalari ishlamaydi ❌ |

👉 Xotira tejash vositalari aynan Windows'da yo'q.

### ✅ Bulut juda arzon

```
RTX 3090 (24 GB):            $0.07–0.21 / soat
F5-TTS 7 soatlik fine-tune:  $1–4
```

8 GB bilan kurashishning ma'nosi yo'q. **Ishlab chiqish 3050 da, o'qitish bulutda.**

**Halol eslatma:** *"Not one of the TTS model cards below publishes a VRAM requirement. We checked."* — yuqoridagi raqamlarning ~80% jamoa hisobotlari.

---

## 5. Narx (agar fine-tune qilinsa)

Nashr etilgan haqiqiy natijalar:

| Kim | Til | Ma'lumot | GPU | Vaqt | Narx |
|---|---|---|---|---|---|
| `lpscr` | Yunon | 40h + 20h ingliz | 1× RTX 3090 | yarim kun | **$2.64** |
| — | Polyak | 90h | 1× A100 80GB | ~24h | $16–38 |
| `jpgallegoar` | Ispan | 218h | 1× RTX 4090 | 6h50m/epoch | — |

Realistik byudjet: 3 urinish + storage + LoRA = **$75–400, maqsad ~$150–250.**

### Ma'lumot talabi (F5-TTS jamoa konsensusi, #143)

```
Bitta ovoz:              10–15 soat
Yangi til, ko'p spiker:  50 soat minimum
                        100 soat yaxshi
                        300+ soat optimal
```

### Fine-tune konfiguratsiyasi — xatolardan saqlanish

```
Base:       pretrained checkpoint'dan BOSHLANG  (noldan — ishlamaydi)
lr:         1e-5     ← 7.5e-5 EMAS (base lr; talaffuzni buzadi)
use_ema:    False    (erta fine-tune'da)
usul:       LoRA     (to'liq SFT bitta spikerda = overfitting)
```

**Muvaffaqiyatsizliklar:**
- Arab, 61 soat, char-level → **overfit**: o'qitish ma'lumotida 85% aniq, ko'rilmagan matnda *"inaccurate and unintelligible"*
- Xitoy, 33 soat, `lr=7.5e-5` → *"pronunciation is becoming increasingly unclear"*
- Bittasi fine-tune checkbox'ni belgilamay **noldan o'qitib qo'ygan** → *"output started as noise"*

**Eng foydali maslahat** (turk TTS blogidan):
> *"A 10-hour dataset with high lexical variety will outperform a 30-hour monotone corpus."*

---

## 6. Qaror

### Ikki maqsad — ularni aralashtirmang

```
┌──────────────────────────────────────────────────────┐
│  A: "MENING ovozimda gapiradigan TTS"                │
│  → 1 odam × 20-25 soat  →  klassik model yoki LoRA   │
│  → ✅ ishlaydi, ❌ boshqa ovozga taqlid qilmaydi      │
└──────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────┐
│  B: "10 sekunddan HAR QANDAY ovozni klonlash"        │
│  → KO'P odam: 50h min / 100h yaxshi / 300h+ optimal  │
│  → yoki: TAYYOR modelni ishlatish (OmniVoice)        │
└──────────────────────────────────────────────────────┘
```

### Tavsiya etilgan ketma-ketlik

| Qadam | Ish | Vaqt | Narx |
|---|---|---|---|
| **1** | **OmniVoice'ni sinash** — 10 sek o'zbek referens, `language_id="uz"`, Kotib bilan baholash | **1 kun** | **$0** |
| 2 | Matn normalizatsiya qatlami (`uztext` apostrof + `uztextnorm` raqam) | 1 hafta | $0 |
| 3 | Agar sifat yetarli bo'lsa — **TAMOM**. Nokommersiya uchun ishlatiladi | — | $0 |
| 4 | Tijorat kerak bo'lsa → **Chatterbox (MIT)** + Abduqayum yo'lini takrorlash | 2–4 hafta | $5–50 |
| 5 | Sifat yetmasa → LoRA, filtrlangan **UzbekVoice** bilan | 2–4 hafta | $50–250 |

### 1-qadam nega hal qiluvchi

**$0 va 1 kun.** Agar OmniVoice yetarli bo'lsa — qolgan hamma ish kerak emas. Buni bilmasdan oyni rejalashtirish — bekor mehnat.

### `ai dataset maker` ning yangi vazifasi

Loyiha bekor bo'lmaydi, **maqsadi aniqlashadi**: bitta kitobni kesish emas, balki **mavjud o'zbek ASR korpuslarini TTS sifatiga tozalash.**

Eng katta imkoniyat: **UzbekVoice (~900–1,200 soat) OmniVoice'ning 50 datasetlik ro'yxatida YO'Q.** Ya'ni uni tozalash — sifatni haqiqatan oshira oladigan yagona richag.

---

## 7. Yopilmagan savollar (halol ro'yxat)

| Savol | Holat |
|---|---|
| OmniVoice'ning o'zbek 115 soati qaysi korpusdan? | **Noma'lum** — USC ham, UzbekVoice ham uning ro'yxatida yo'q |
| USC sample rate | **Ko'rsatilmagan** — Telegram orqali yig'ilgan, transkodlangan bo'lishi mumkin |
| USC litsenziyasi | **Ziddiyat** — ISSAI CC-BY-4.0 vs HF MIT |
| mohir.ai | DNS timeout — tekshirilmadi |
| tilmoch.ai / Tahrirchi | HTTP 403 — tekshirilmadi |
| Abduqayum modelining sifati | Metrika **nashr etilmagan**, faqat demo |
| Windows desktop VRAM iste'moli | O'lchangan raqam **topilmadi** |
| TTS Arena V2, VibeVoice-Large kartasi | HTTP 401 — olinmadi |

Bu joylar taxmin bilan to'ldirilmagan.

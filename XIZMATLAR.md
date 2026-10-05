# Nega avtomatik dataset tayyorlash xizmati yo'q?

**Savol:** avtomatik dataset tayyorlab beradigan xizmatlar nega yo'q? Bo'lsa, ular qanday ishlaydi?

**Sana:** 2026-10-01
**Batafsil:** `tadqiqot/xizmatlar/` — 3 fayl (tijorat_xizmatlar, mavjud_pipelinelar, nega_yoq), 185 KB

---

## 0. Qisqa javob

1. **Bor.** Azure Custom Neural Voice aynan shu ishni qiladi va natijani yuklab olishga beradi.
2. **16 ta nashr etilgan pipeline** mavjud, **8 tasining kodi ochiq**. "Hech kim qilmagan" degan taxmin — noto'g'ri.
3. **Lekin:** Azure yopiq va o'zbek yo'q; qolgan xizmatlar datasetni **qaytarmaydi**; ochiq pipelinelar **Windows'da ishlamaydi** va o'zbek qatlami yo'q.
4. **Nega mahsulot emas:** avtomatlashtiriladigan qism bepul, avtomatlashtirilmaydigan qism 1 soatga **30 odam-soat** turadi, xaridor scrape qiladi, va huquqiy javobgarlik vositachiga tushadi.
5. **Niche bor:** kam resursli til + o'z ma'lumotini o'zida qoldirish + rozilik/provenance.

---

## 1. Tijorat xizmatlar — bor, lekin yopiq

### Azure Custom Neural Voice — aynan shu ishni qiladi

"Long audio + transcript" turini qabul qiladi va ichida:

```
Uzun audio + skript
      ↓  Batch Transcription API bilan segmentlash
      ↓  Skriptga moslash (yoki ASR bilan transkript yasash)
      ↓  Har utterance ga unikal ID
      ↓  Talaffuz bahosi 0–100  →  70 dan past = AVTOMATIK RAD
      ↓  SNR: 20 dB kesish, 35 dB maqsad
      ↓  100 ms sukunatlarni avtomatik to'g'rilash
      ↓  Raqam/qisqartma normalizatsiyasi
      ↓  .mp3 → .wav 24 kHz
      ✅ NATIJANI YUKLAB OLISHGA BERADI
```

Rasmiy hujjatdan: *"only the segmented utterances and their mapping transcripts are stored for **your downloading**"*

### Lekin 3 to'siq

| To'siq | Tafsilot |
|---|---|
| 🔒 **Yopiq** | *"only customers managed by Microsoft... working directly with Microsoft account teams"* — yakka dasturchi kirmaydi |
| 🌍 **O'zbek yo'q** | Segmentatsiya faqat 14 lokalda (`zh en fr de hi it ja pt-BR es`...). `uz-UZ` da Azure'da TTS ovozi **umuman yo'q** (ASR bor) |
| 💰 **Bog'langan sotuv** | Training **$52/compute-hour** (~10 soat ≈ $520), hosting **$4.032/soat ≈ $2,943/oy**. Alohida "dataset tayyorlash" SKU si **butun bozorda yo'q** |

### Qolgan hamma xizmat datasetni qaytarmaydi

ElevenLabs ochiq yozgan:
> *"You cannot download or export your voice clones as standalone files."*

Google, Amazon Polly, Resemble, Cartesia, Murf, Speechify — **hech biri** bermaydi.

### 💡 Nega — asl sabab

**Dataset bu moat.**
```
Tozalangan korpusni qaytarsang → mijoz XTTS/F5 o'qitadi → va KETADI
```

Ikkinchi sabab — **verifikatsiya zanjiri**: Azure biometrik speaker recognition, Google majburiy consent jumlasi, Speechify challenge phrase, ElevenLabs mikrofon captcha. Bu "ovozni ber, datasetni olib ket" modeli bilan sig'ishmaydi.

### O'zbek uchun yagona tijorat yo'l

```
ElevenLabs PVC (Professional Voice Cloning)
  Eleven v4 da  →  uzb  ✅
  Minimum       →  30 daqiqa   (sizda 54 daqiqa bor)
  Tavsiya       →  2–3 soat
  Fine-tune     →  3–6 soat kutish
  Tarif         →  Creator
```
Shartlar: faqat **o'z ovozingiz** (*"Even with their consent, you cannot clone someone else's voice"*), dataset qaytmaydi, model ularning akkauntida.

Boshqalar: Google ICV (10 sek, allowlist, **o'zbek yo'q**), Polly 42 til (**o'zbek yo'q**), Murf (30 sek instant / 90 daqiqa professional, 1–4 hafta, faqat Enterprise, **o'zbek yo'q**).

### Ma'lumot bozoridagi yagona ochiq narx

```
Defined.ai Data Access Plan
    315 soat  →  €50,000
  3,125 soat  →  €500,000
             ────────────
             ≈ €160 / soat
```

Appen "expressive TTS annotation" ni eslatadi, narx bermaydi. Shaip, Sama, Scale, Summa Linguae — **ma'lumot topilmadi**.

---

## 2. Ochiq pipelinelar — 16 ta bor, 8 tasining kodi ochiq

**"Hech kim qilmagan" taxmini noto'g'ri.** Lekin foydali xulosa saqlanadi: **hech kim buni sizning tilingiz uchun, sizning OS'ingizda qilmagan.**

### 🎯 8 ta mustaqil laboratoriya BIR XIL ketma-ketlikka keldi

Bu eng qimmatli topilma — agar 8 laboratoriya mustaqil ravishda bir xil dizaynga kelgan bo'lsa, demak **bu to'g'ri dizayn**:

```
1.  standartlashtirish (format, rate, mono)
2.  [enhance / denoise]              ← ixtiyoriy
3.  diarizatsiya (kim gapiryapti)
4.  Silero-VAD bilan 3–30 s ga kesish
5.  ASR                               ← ⚠️ BO'G'IZ, pastda
6.  [referens matnga alignment]       ← sizda matn bor, shu qadam
7.  matn normalizatsiya + punktuatsiya tiklash
8.  sifat ko'rsatkichlarini hisoblash
9.  chegara / tier bo'yicha ajratish
10. manifest
```

Har bosqichda **de-fakto standart vosita** ham aniqlangan: Silero VAD, pyannote, Whisper variantlari, DNSMOS P.835 OVRL, CTC forced alignment.

### Nashr etilgan chegaralar

| Pipeline | Chegaralar |
|---|---|
| Emilia | DNSMOS ≥ 3.0 |
| WenetSpeech4TTS | 4.0 / 3.8 / 3.6 (ichma-ich: 945h / 4,056h / 7,226h) |
| IndicVoices-R | C50 ≥ 30 dB + SNR ≥ 25 dB + CER ≤ 5% |
| GLOBE | SNR ≥ 0 dB + bandwidth ≥ 12 kHz + edit-distance ≤ 1 |
| MMS | CER ≤ 10% + score ≥ −0.2 |
| EuroSpeech | CER < 10 / 20 / 30% |

### Saqlanish foizi — bir joyga to'planadi

| Pipeline | Qancha qoldi |
|---|---|
| GLOBE | **16%** (3,347h → 535h) |
| Emilia | **29.43%** (598.87h → 176.22h) |
| WenetSpeech4TTS | ~56% |
| Granary | 60.7% |
| EuroSpeech | 65.4% |
| HiFiTTS-2 | 82% |
| Libriheavy | ~83% |
| LibriTTS-R | **100%** (filtrlash emas, **tiklash**) |

⚠️ **Emilia ziddiyati tasdiqlandi:** 29.43% raqami ~600 soatlik namunadan, DNSMOS ≥ 3.0 da. `arXiv:2501.15907` esa Emilia-Large (216,313h) uchun chegarani **jimgina 2.4 ga tushirgan** va natijadagi foizni **hech qachon e'lon qilmagan**.

### 🚩 Uchta jiddiy ogohlantirish

| Pipeline | Muammo |
|---|---|
| IndicVoices-R, GLOBE | Dataset chiqarilgan, lekin **pipeline kodi YO'Q** — faqat retsept |
| LibriTTS-R | Miipher vaznlari chiqarilgani aytilmagan → **eng yaxshi bosqichi takrorlanmaydi** |
| Libriheavy | `download/librilight` yo'li **hardcode** qilingan |
| EuroSpeech | Algoritmik jihatdan **eng mos**, lekin litsenziya **"TBD"** va **4 commit** |
| WorldSpeech | **2 commit**, EuroSpeech repo'siga havola qiladi |
| NeMo SDP | Eng universal, lekin **Windows native yo'q** — WSL2/Docker |

### ⚠️ Haqiqiy bo'shliq — 5-bosqich, ASR

EuroSpeech va WorldSpeech **ikkisi ham ochiq aytadi**: hosil *tayyor ASR sifati bilan cheklangan*.

```
Whisper o'zbek WER = 90.2%
      ↓
Qolgan TO'QQIZ bosqich ahamiyatsiz
```

👉 Bu sizning **Kotib tanlovingizni tasdiqlaydi.** Loyihaning eng muhim qarori allaqachon to'g'ri qilingan.

### 🎯 Ikki o'zbekka xos topilma

**1. NeMo SDP'da o'zbek konfiguratsiyalari ALLAQACHON bor:**
```
dataset_configs/uzbek/fleurs
dataset_configs/uzbek/mcv
dataset_configs/uzbek/uzbekvoice      ← UzbekVoice uchun!
```

**2. WorldSpeech'da `uz_uz` bor — lekin:**
```
33.7 soat        ← juda kam
DNSMOS OVR 2.84  ← Emilia'ning O'Z chegarasidan past
RFE/RL           ← nokommersiya
```

### 🔥 Eng qimmatli texnik topilma

**WorldSpeech'ning iterativ qayta-alignment usuli** — zaif ASR'li tillar uchun **+19.5% dan +201.1% gacha** qo'shimcha ma'lumot bergan.

👉 Bu **aynan o'zbek tilining holati.** Bir marta alignment qilib tashlamaslik, balki takrorlab yaxshilash.

### Nimaga qurish kerak

```
Lhotse                              ← konteyner (ma'lumot modeli)
  + Emilia-Pipe bosqich tartibi     ← blueprint (config.json har papkani qabul qiladi)
  + KOTIB 5-bosqichda               ← Whisper emas!
  + ctc-forced-aligner              ← 6-bosqich
  + EuroSpeech ikki bosqichli CER refinement (±15 so'z)
  + WorldSpeech iterativ qayta-alignment   ← +19.5–201% ma'lumot
```

---

## 3. Nega bu biznes bo'lmagan — 5 sabab

### 1. Avtomatlashtiriladigan qism bepul

```
ASR bugun:  1 soat audio → $0.15–0.36   (AssemblyAI, Whisper API)
```

**Emilia-Pipe** — mualliflarining o'z so'zi bilan *"the first open-source preprocessing pipeline"* — xom audioni tayyor datasetga aylantiradi, va u bilan **101,000+ soat** chiqarilgan (arXiv:2407.05361, SLT 2024).

👉 Sotmoqchi bo'lgan mahsulot allaqachon GitHub'da.

### 2. Avtomatlashtirilmaydigan qism halokatli qimmat

Bambara tilida dala tadqiqoti (arXiv:2510.12781, 10 ona tili sohibi, 53 soat):

```
ASR chiqishini TUZATISH:
  1 audio-soatga → 30 odam-soat (lab)
                 → 36 odam-soat (dala)

Ingliz tilida (ASR 95%) → ~5 daqiqa/soat
```

👉 Muammo **faqat kam resursli tilda** bor — va aynan o'sha yerda avtomatlashtirish ishlamaydi.

### 3. Xaridor yo'q

| Kim | Soat |
|---|---|
| Whisper | 680,000 (scrape) |
| Google USM | **12,000,000** |
| Meta MMS | 1,100+ til (diniy matn o'qishlaridan) |
| Eng katta vendor paketi | **3,125 soat / €500,000** |

Vendor paketi Whisper korpusining **0.46%**i. O'sha korpusni vendor narxida olish **€108.8M** bo'lardi.

👉 Katta laboratoriyalar sotib olmaydi — **scrape qiladi.**

### 4. ⚠️ Huquqiy javobgarlik vositachiga tushadi

**2026-05-19, Chicago federal sudi — 9 ta class action:**

| | |
|---|---|
| Kim | Audiokitob o'quvchilari (Dorcus, Nassif) + jurnalistlar |
| Kimga qarshi | Google, Apple, Microsoft, Meta, **ElevenLabs**, NVIDIA |
| Asos | BIPA + right of publicity |

Google ishidagi aybnoma: **"long-form, single-speaker, studio-quality"** audio scrape qilingani.

| Qonun | Nima qiladi |
|---|---|
| **BIPA** (Illinois) | $1,000 / $5,000 **har buzilish uchun**, real zarar shart emas (*Rosenbach*, 2019) |
| **ELVIS Act** (Tennessee) | Javobgarlikni **vosita sotuvchiga ham** yuklaydi |
| *Lehrman v. Lovo* (S.D.N.Y., 2025-07-10) | **Copyright ovoz imitatsiyasini himoya qilmaydi** |

Oxirgi nuqta nozik: copyright himoya qilmasa, TDM istisnolari (EU DSM Art.4, Yaponiya 30-4) ham **qutqarmaydi** — biometrik va publicity huquqi copyright'dan **mustaqil**.

### 🟢 Sizning holatingiz boshqa — muhim farq

Sudlar **ruxsatsiz** olingan ovozlar haqida:

| Manba | Holat |
|---|---|
| **O'z ovozingiz** (54 daqiqa) | ✅ Rozilik sizda |
| **FeruzaSpeech** | ✅ CC BY 4.0 |
| **Common Voice uz** | ✅ CC-0 |
| **USC** | 🟡 Litsenziya ziddiyati (CC-BY-4.0 vs MIT) — aniqlash kerak |
| **UzbekVoice** | 🟡 Litsenziya ko'rsatilmagan — aniqlash kerak |
| Internetdan audiokitob yig'ish | 🔴 **Aynan sudga tortilgan narsa** |

### 5. Urinib ko'rganlar yopildi

| Kim | Nima bo'ldi |
|---|---|
| **cielo24** | 2012-dan, ~70 xodim, VC puli, 10 yillik time-aligned korpus → **yopildi**, ma'lumot aktivlari "mid-six figures"ga sotildi |
| **15.ai** | Dataseti 4chan ko'ngillilari bepul qilgan → 2022-09 da huquqiy sabablar bilan **o'chdi** |
| **Defined.ai** | Yopilmadi, lekin 2022-da sof data provider'dan platforma+evaluation'ga **pivot** |

---

## 4. Niche qayerda

### (1) Pul alignment'da emas, ROZILIKDA

`aipodcast.io` sotadigan narsa:
> **"verified consent + commercial training rights + 100% guest re-clearance"**

Texnika emas — **huquqiy tozalik**. Yuqoridagi 9 sud ishi aynan shu bozorni yaratdi.

### (2) On-prem / BYOR tooling

Audio foydalanuvchi kompyuteridan **chiqmaydi**. GDPR Art.9 DPIA va BIPA argumentini hal qiladi. Bulut xizmatlari buni qila olmaydi.

### (3) Kam resursli til vertikali

```
Ochiq o'zbek nutqi:  ≈430 soat
   (USC 105 + Common Voice 266 + FeruzaSpeech 60)
Whisper ingliz:      563,000 soat
                   ─────────────
Nisbat:                1 : 1,300
```

Moliyalashtiruvchilar: Gates → Masakhane **$4,998,785**; IDRC **CAD 5.36M**; RESPIN (10,000+ soat); Lacuna Fund.

⚠️ Lekin ular **grant mantig'ida** ishlaydi va natija **ochiq** bo'lishini talab qiladi — xizmat marjasi, mahsulot marjasi emas.

---

## 5. 🎯 Sizning yagona haqiqiy moat'ingiz

> **Lotin/kirill ikki tomonlama transkripsiya va o'zbek normalizatsiyasi — Emilia-Pipe'da YO'Q.**

| Qism | Holat |
|---|---|
| Umumiy pipeline (10 bosqich) | ✅ Tayyor, bepul, qaytarib qurish bekor |
| ASR 5-bosqichda | ✅ Kotib — sizda bor, Whisper'dan 13× yaxshi |
| NeMo SDP o'zbek konfiglari | ✅ Allaqachon mavjud |
| **O'zbek matn qatlami** | 🔴 **Hech kimda yo'q** |
| **Iterativ qayta-alignment** | 🟡 WorldSpeech'da bor (2 commit), ko'chirib olish kerak |

O'zbek qatlamining ishi: 12 ta o'xshash apostrof kanonlashtirish, digraflar (`sh ch ng`), lotin↔kirill, raqam/sana/qisqartma, 2026 alifbo islohoti (`Ö Ğ Ş Ç`).

👉 **Qurish kerak bo'lgani — butun pipeline emas, o'zbek qatlami.**

---

## 6. Yopilmagan savollar

| Savol | Holat |
|---|---|
| Emilia-Large saqlanish foizi | **E'lon qilinmagan** (chegara jimgina 2.4 ga tushirilgan) |
| BIPA statut matni | ma'lumot topilmadi |
| ELVIS Act kuchga kirish sanasi | **Manbalar qarama-qarshi** |
| SAG-AFTRA stavkalari | ma'lumot topilmadi |
| Appen / Google shartnomasi | ma'lumot topilmadi |
| Shaip, Sama, Scale, Summa narxlari | ma'lumot topilmadi |
| LibriTTS-R Miipher vaznlari | Chiqarilgani **aytilmagan** |
| EuroSpeech litsenziyasi | **"TBD"** |

`nega_yoq.md` ning 9-bo'limida jami **14 ta** tasdiqlanmagan punkt ro'yxatlangan. Taxmin bilan to'ldirilmagan.

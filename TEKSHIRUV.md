# Chuqur tekshiruv — hisobot va tasdiqlash uchun takliflar

**Sana:** 2026-10-02
**Qoida:** bu hujjatda HECH NARSA O'ZGARTIRILMAGAN. Faqat tekshiruv natijalari va
takliflar. Har bir taklif siz tasdiqlagandan keyin bajariladi.

---

## 0. Qisqa xulosa

| | Soni |
|---|---|
| Tekshirilgan funksiya/qaror | 14 |
| ✅ To'g'ri ishlayotgani tasdiqlandi | 6 |
| 🔴 Jiddiy muammo (maqsadga yetishga to'sqinlik qiladi) | 2 |
| 🟡 Nomuvofiqlik / sifat yo'qotish | 3 |
| 🔵 Dizayn qarori kerak (to'g'ri/noto'g'ri emas, tanlov) | 3 |
| Boshqalar qiladigan, bizda yo'q | 8 |

**Eng muhimi:** hozirgi tizim 10 daqiqada ishlaydi, lekin **20 soatlik kitobda
ikki joyda yiqiladi** — xotira (27.6 GB kerak, 15.9 bor) va vaqt (~9 soat ishlov).
Ikkisi ham hal qilinadigan, lekin kod o'zgarishi kerak.

---

## 1. ✅ To'g'ri ishlayotgani TASDIQLANDI

| # | Nima tekshirildi | Dalil |
|---|---|---|
| 1 | **Score — ehtimollik (0..1), log emas** | `torchaudio/.../aligner.py:41` — `scores = scores.exp()  # convert back to probability`. Chegaralarim (0.35, 0.02) to'g'ri shkalada |
| 2 | **Chunking siljish bermaydi** | chunk 10/20/40 s → 1-so'z 0.579/0.579/0.580 s, oxirgi 604.98 s hamma holatda |
| 3 | **Kotib'ga `language=uz` berish zararsiz** | `generation_config.json` da allaqachon `language: uz, task: transcribe` bor — mening kwarg'im takroriy, lekin zararsiz |
| 4 | **MMS_FA vocab o'zbekni qamraydi** | 29 token: `a-z ' * -`. `ʻ`→`'` xaritalash ishlaydi (977/977 so'z) |
| 5 | **`*` wildcard tokeni mavjud** | vocab'da bor, `with_star=True` ishlaydi |
| 6 | **Defisli so'zlar buzilmaydi** | `so'z-birikma` → align `so'zbirikma` (bitta token, CTC uzluksiz belgilar bilan ishlaydi) |

---

## 2. 🔴 JIDDIY — maqsadga to'sqinlik qiladi

### 2.1 Xotira: uzun fayl yuklanmaydi

**Dalil (hisoblangan):**
```
load_audio()  →  torchaudio.load()  →  BUTUN faylni xotiraga oladi

 1 soat, 48 kHz stereo manba  →   1.4 GB   ✅
 5 soat                       →   6.9 GB   ✅
20 soat                       →  27.6 GB   ❌  RAM 15.9 GB
```
Tadqiqot aynan buni ogohlantirgan edi (`librosa.load` 10 soatda ~14 GB). Men
`torchaudio.load` ishlatdim — xuddi shu muammo.

**Oqibat:** 20 soatlik kitob — loyihaning asl maqsadi — **ishlamaydi**.

**Taklif:** oqimli yuklash. `soundfile.blocks()` yoki `ffmpeg -ss/-t` bilan
bo'lak-bo'lak o'qish, 16 kHz mono ga o'girib diskka yozish, keyin shundan ishlash.
Alignment allaqachon bo'laklab ishlaydi — faqat yuklash o'zgaradi.

**Xavf:** past. Alignment mantiqiga tegilmaydi.
**Hajm:** ~40-60 qator, `align.py` + `main.py`.

### 2.2 Vaqt: ASR back-check 86% vaqtni yeydi

**Dalil (o'lchangan, 10 daqiqa audio):**
```
alignment       16 s
ASR back-check 231 s   ← 86%
qolgani        ~21 s
```
Ekstrapolyatsiya: **20 soat audio → ~8.9 soat ishlov, shundan 7.7 soat ASR.**

**Sabab:** Kotib — Whisper **large** arxitekturasi (d_model 1024, 24 qatlam,
~1.5B parametr). `transformers.pipeline` orqali avtoregressiv generatsiya.

**Taklif (tadqiqotdan):** `faster-whisper` (CTranslate2). Tadqiqot 3-4× tezlik
deydi. Kotib Whisper formatida → `ct2-transformers-converter` bilan o'giriladi.
int8 kvantlash → VRAM ham kamayadi (batch oshadi).

**Xavf:** o'rta. Yangi dependency; Windows'da CTranslate2 wheel bor, lekin
Kotib konversiyasi sinalmagan. Aniqlik o'zgarishi mumkin (int8).
**Hajm:** `quality.py` da `KotibASR` klassini almashtirish, ~50 qator + konversiya.

**Alternativa (xavfsizroq):** hozirgi pipeline'da `batch` ni 4→8 ga oshirish
(expandable_segments bilan). Taxminan 1.5× yutuq, lekin 3-4× emas.

---

## 3. 🟡 NOMUVOFIQLIK — sifat yo'qotish

### 3.1 `trim_and_pad` `PAD_START` ni bekor qiladi

**Dalil (o'lchangan, eksport qilingan 30 klip):**
```
Segmentda qo'shildi:  PAD_START = 350 ms
Eksportda topildi:    boshida jimlik median 3 ms (min 0)
                      oxirida jimlik median 114 ms   ← 100 ms pad + qoldiq
```
`export.trim_and_pad()` chetlardagi jimlikni **1% chegarada** kesadi, keyin faqat
**oxiriga** 100 ms qo'shadi. Ya'ni boshiga qo'shgan 350 ms yana olib tashlanadi.

**Ikki muammo bir vaqtda:**
1. Pad'ni qo'shish-olib tashlash — bir-biriga qarshi ishlayotgan ikki bosqich
2. Bosh/oxir nomutanosib: oxirida 100 ms bor, boshida 0. Tadqiqot (StyleTTS2):
   *"pad miqdori emas, MUTANOSIBLIGI muhim"*

**Nega xavfli:** 1% chegara **klipga nisbatan**. Jim boshlangan so'z (`u`, `i`)
1% dan past bo'lsa — kesiladi. Aynan `PAD_START` qoplamoqchi bo'lgan muammo
eksportda qaytib keladi.

**Taklif:** `trim_and_pad` ni o'zgartirish — boshiga ham bir xil 100 ms pad
(StyleTTS2 uslubi), yoki trim'ni butunlay olib tashlab segmenter chegaralariga
ishonish. Ikkisidan birini **siz tanlaysiz** (3.4-bo'limga qarang).

**Xavf:** past. Faqat `export.py`.

### 3.2 Katta harflar yo'qoladi

**Dalil (o'lchangan):**
```
'Toshkent shahri. Alisher Navoiy.'  →  'toshkent shahri. alisher navoiy.'
"O'ZBEKISTON Respublikasi"           →  "o'zbekiston respublikasi"
```
`uztextnorm.normalize()` hammasini kichik harfga o'giradi. Men buni to'xtatmaganman.

**Nega muhim:** Libriheavy **ataylab** kitobdagi katta harf va punktuatsiyani
saqlaydi. Zamonaviy TTS (XTTS, F5, OmniVoice) **case-sensitive** — "Navoiy"
(ism) va "navoiy" (so'z) prosodiyasi farq qiladi. Gap boshi katta harfi
intonatsiya signali.

**Taklif:** uztextnorm'ni faqat **raqam/sana bo'laklariga** qo'llash, qolgan
matnga tegmaslik. Yoki normalizatsiyadan keyin asl matndan harflarni qayta
tiklash (character-level diff).

**Xavf:** o'rta. uztextnorm ichki ishlashini o'zgartirmaslik kerak — uni
o'rash (wrap) qilinadi. Sinov kerak.

### 3.3 So'z bali — tenglashtirilmagan o'rtacha

**Dalil:** `align.py` da so'z bali = token ballarining **oddiy o'rtachasi**.
torchaudio tutoriali **uzunlik bo'yicha tenglashtirilgan** o'rtacha ishlatadi
(uzun token ko'proq hissa qo'shadi).

**Oqibat:** kichik. Qisqa `'` yoki bitta harf tokeni uzun bo'g'in bilan teng
vazn oladi. Chegaralar shu usulda kalibrlangan — o'zgartirsak qayta kalibrlash kerak.

**Taklif:** hozircha **tegmaslik**. Agar 3.4 dagi boshqa o'zgarishlar uchun
qayta kalibrlash qilinsa — shunda birga tuzatish.

---

## 4. 🔵 DIZAYN QARORLARI — siz tanlaysiz

Bular xato emas. Javob maqsadingizga bog'liq.

### 4.1 Apostrof: `ʻ` (U+02BB) yoki `'` (ASCII)?

Hozir dataset matnida **`ʻ`** (uztext kanonik shakli). Lekin:

| Maqsad model | Kutadi |
|---|---|
| O'z VITS (char tokenizer) | ixtiyoriy — nima bersangiz o'rganadi |
| Chatterbox (MIT) | 2454-token grapheme tokenizer, o'zbek lotini "without modification" — qaysi apostrof, **tekshirilmagan** |
| OmniVoice | Qwen3 tokenizer — `ʻ` ni bilishi ehtimol, lekin **tekshirilmagan** |
| Kotib (CER uchun) | farqi yo'q — ikkisi ham romanizatsiya qilinadi |

**Qaror:** maqsad model aniq bo'lgach tanlash. Hozircha `ʻ` — chunki
o'zbek standarti va `uztext` qaytara oladi.

### 4.2 Bosh pad: 0 yoki 100 ms?

3.1 ga bog'liq. Variantlar:

| Variant | Afzallik | Kamchilik |
|---|---|---|
| A: boshi 0, oxiri 100 ms (hozirgi) | LJSpeech'ga yaqin | Nomutanosib, PAD_START bekor bo'ladi |
| B: ikkisi ham 100 ms | StyleTTS2 tavsiyasi, mutanosib | Boshida ozgina jimlik |
| C: trim yo'q, segmenter chegaralari | PAD_START ishlaydi | Jimlik miqdori klipdan klipga farq qiladi — TTS uchun **yomon** |

**Tavsiya:** B.

### 4.3 `--star edges` yoki `--star between`?

Hozir `edges` (faqat boshi/oxiri). Kitob o'qishda o'quvchi so'z tashlab ketsa,
`between` (har so'z orasida wildcard) uni yutadi. Lekin `between` aligner'ga
"so'zni o'tkazib yuborish" erkinligini beradi — xatolarni **yashirishi** mumkin.

**Tavsiya:** sinab ko'rish — FLEURS testida ikkalasini solishtirish (saqlanish
va CER). 10 daqiqa, kod o'zgarmaydi (CLI parametri).

---

## 5. BOSHQALAR QILADIGAN, BIZDA YO'Q

Tadqiqotdan (`tadqiqot/xizmatlar/mavjud_pipelinelar.md`), ustuvorlik bo'yicha:

| # | Nima | Kim qiladi | Bizga ta'siri | Hajm |
|---|---|---|---|---|
| 1 | **Iterativ qayta-alignment** | WorldSpeech | Zaif ASR tillarda **+19.5–201%** ma'lumot. O'zbek aynan shu holat | o'rta |
| 2 | **Oqimli audio** | NeMo SDP, Lhotse | 2.1 ni hal qiladi | kichik |
| 3 | **faster-whisper** | ko'pchilik | 2.2 ni hal qiladi | o'rta |
| 4 | **Katta harf saqlash** | Libriheavy | 3.2 ni hal qiladi | o'rta |
| 5 | **Ikki bosqichli CER refinement (±15 so'z)** | EuroSpeech | Chegara xatolarini mahalliy tuzatish | o'rta |
| 6 | **DNSMOS / UTMOS** | Emilia, WenetSpeech4TTS | Shovqinli klipni SNR'dan yaxshi ushlaydi. Mening SNR'im WADA emas | kichik |
| 7 | **Denoise + MCD ≤ 5 dB** | LibriTTS-R | MOS 2.59 → 4.06. Lekin shovqinsiz studiya yozuvida kerak emas | katta |
| 8 | **Manifest + resume** | NeMo SDP | 9 soatlik ish yarmida to'xtasa qaytadan boshlamaslik | o'rta |
| 9 | **Review UI** | (tadqiqot tavsiyasi) | "ko'rish" segmentlarni qo'lda tekshirish — hozir 5 ta kutib turibdi | katta |

---

## 6. STRATEGIK ESLATMA — hali qilinmagan

Tadqiqotning **#1 tavsiyasi**: *"OmniVoice'ni sinash — 1 kun, $0. Yetarli bo'lsa
qolgan ish kerak emas."*

Biz dataset maker'ni qurdik, lekin **OmniVoice sinalmadi**. Agar u sizning
ovozingizda yaxshi gapirsa — dataset maker'ning vazifasi o'zgaradi (UzbekVoice
tozalash), siz o'qib yozishingiz kerak bo'lmaydi.

Bu 1-2 soatlik ish: `pip install omnivoice`, 10 sek ovoz, o'zbek matn.
Hech qanday kod o'zgarmaydi.

---

## 7. YO'L TANLOVI

### Yo'l A — hozirgini sayqallash (1-2 kun)
2.1 + 3.1 + 4.2(B) → uzun fayl ishlaydi, pad mutanosib. ASR hali sekin.
**Natija:** 20 soatlik kitob 9 soatda qayta ishlanadi, bir kechada.

### Yo'l B — to'liq tezlashtirish (3-4 kun)
A + 2.2 (faster-whisper) + 3.2 (katta harf) + 5.1 (iterativ alignment).
**Natija:** 20 soat → ~3 soat, +20-200% ma'lumot, sifatli matn.
**Xavf:** faster-whisper konversiyasi sinalmagan.

### Yo'l C — avval OmniVoice (2 soat), keyin qaror
6-bo'lim. Natijaga qarab A yoki B, yoki umuman boshqa yo'nalish.
**Tavsiya etiladi** — eng arzon ma'lumot.

---

## 8. TASDIQLASH RO'YXATI

Har birini alohida tasdiqlang (raqam bilan). Tasdiqlanmagani qilinmaydi.

| # | Ish | Hajm | Xavf | Bo'lim |
|---|---|---|---|---|
| **1** | OmniVoice'ni o'z ovozingizda sinash | 2 soat | yo'q | 6 |
| **2** | Oqimli audio yuklash (20 soat ishlasin) | kichik | past | 2.1 |
| **3** | Eksport pad'ini mutanosib qilish (B varianti) | kichik | past | 3.1, 4.2 |
| **4** | `--star edges` vs `between` ni FLEURS'da solishtirish | 10 daq | yo'q | 4.3 |
| **5** | faster-whisper ga o'tish | o'rta | o'rta | 2.2 |
| **6** | Katta harflarni saqlash | o'rta | o'rta | 3.2 |
| **7** | Iterativ qayta-alignment (WorldSpeech) | o'rta | o'rta | 5.1 |
| **8** | DNSMOS/UTMOS qo'shish | kichik | past | 5.6 |
| **9** | Manifest + resume | o'rta | past | 5.8 |
| 10 | So'z balini tenglashtirish + qayta kalibrlash | kichik | past | 3.3 |
| 11 | Review UI | katta | past | 5.9 |
| 12 | Denoise + MCD | katta | o'rta | 5.7 |

**Mening tavsiyam:** `1 → 2 → 3 → 4` (hammasi 1 kun ichida, past xavf), keyin
natijaga qarab 5-7.

---

## 9. Tekshirilmagan — halol ro'yxat

| Nima | Nega |
|---|---|
| Chatterbox/OmniVoice tokenizer'i `ʻ` ni qanday qabul qiladi | Modelni yuklash kerak (1-2 GB), ruxsat so'ramadim |
| `--star between` natijasi | Ishga tushirish kerak — ruxsat kutyapman |
| faster-whisper'da Kotib aniqligi | Konversiya qilish kerak — o'zgarish, ruxsat kutyapman |
| Bitta spikerli uzun o'qishda saqlanish foizi | Material yo'q — FLEURS ko'p spikerli |
| Kesilgan 3 ms boshdagi jimlik haqiqatan jimlikmi yoki jim so'z boshi | Eshitish kerak — avtomatik ajratib bo'lmadi |

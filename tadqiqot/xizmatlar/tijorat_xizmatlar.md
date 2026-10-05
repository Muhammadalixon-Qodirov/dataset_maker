# Tijorat "custom voice" / ovoz klonlash xizmatlari — TTS dataset quvuri ichdan qanday ishlaydi

**Tadqiqot sanasi:** 2026-10-01 (barcha narxlar shu kunda o'qilgan)
**Savol:** "Agar TTS uchun avtomatik dataset tayyorlash mumkin bo'lsa, nega hech kim buni xizmat qilib qurmagan? Yoki qurganmi — qanday ishlaydi?"
**Bu hujjatning yo'nagi:** faqat tijorat "custom voice" / voice-cloning mahsulotlari. Bu xizmatlar mijozning yozuvlarini qabul qilib, ishlaydigan ovoz modeli qaytaradi — ya'ni foydalanuvchi tasavvur qilgan dataset quvurini **ichda** aynan ishlatadi. Akademik quvurlar va huquqiy/iqtisodiy to'siqlar boshqa agentlar yo'nagida.

> **Metodologiya qaydi:** har bir raqam uchun manba URL ko'rsatilgan. Rasmiy hujjat/narx sahifasidan tasdiqlanmagan raqamlar **"[ikkilamchi manba]"** yoki **"ma'lumot topilmadi"** deb belgilangan. Narx sahifalari o'zgarib turadi — o'qish sanasi 2026-10-01.

---

## 0. QISQA JAVOB (sabrsizlar uchun)

**Ha, bitta xizmat aynan foydalanuvchi so'ragan ishni qiladi: Microsoft Azure AI Speech.** U "Long audio + transcript" (uzun audio + skript) yoki "Audio only" turini qabul qiladi, Batch Transcription bilan segmentlaydi, matnni audioga moslaydi, har bir utterance uchun talaffuz bahosi (0–100) va SNR hisoblaydi, sukunatlarni avtomatik to'g'rilaydi, matnni normallashtiradi — **va natijani yuklab olish mumkin**:

> "After the segmentation is done, only the segmented utterances and their mapping transcripts are stored **for your downloading** and fine-tuning."
> — <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-custom-voice-training-data>

**Lekin uchta devor bor:** (1) Limited Access — faqat Microsoft account team bilan to'g'ridan-to'g'ri ishlaydigan "managed" mijozlar; (2) segmentlash faqat 14 lokalda ishlaydi — **o'zbek tili yo'q**; (3) bu dataset quvuri alohida mahsulot emas, fine-tuning oqimining nojo'ya mahsuli — datasetni olish uchun ovoz modelini ham sotib olishga majbursiz.

Qolgan hamma xizmat (ElevenLabs, Google, Polly, Resemble, Cartesia, Murf, Speechify...) datasetni **ichda ushlab qoladi va faqat qora qutи ovoz qaytaradi**. ElevenLabs buni to'g'ridan-to'g'ri yozib qo'ygan: "You cannot download or export your voice clones as standalone files."

---

## 1. Microsoft Azure AI Speech — Custom Voice (eng batafsil hujjatlangan)

Azure hujjatlari bu sohada eng ochiq — ular dataset quvurining har bir bosqichini e'lon qilgan. Shu sababli bu bo'lim eng uzun: foydalanuvchi quradigan narsaning sanoat standarti aynan shu yerda yozilgan.

### 1.1 Tiers / darajalar

Azure nomenklaturasi 2024–2026 orasida o'zgardi. Eski nomlar: **CNV Lite** va **CNV Pro**. Yangi hujjatlarda: **personal voice** va **professional voice (fine-tuning)**.

| Daraja | Kerakli ma'lumot | Maqsad | Holat |
|---|---|---|---|
| **Custom neural voice lite** (CNV Lite) | **20–50 ta ovoz namunasi**, Speech Studio ichida yozib olinadi. Yozish va test skripti **Microsoft tomonidan oldindan belgilangan** | demo va baholash | public preview. "may be deployed and used more broadly only if you apply for and receive full access" |
| **Professional voice** (CNV Pro) | odatda **300 utterance**; unikal persona uchun **kamida 1000, afzali 2000** utterance | brend/personaj ovozi, audio kontent | Limited Access |
| **Personal voice** | **5–90 sekundlik** inson nutqi namunasi | ilovangiz foydalanuvchilari o'z ovozini yaratishi | Limited Access |

Manbalar:
- <https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/speech-service/text-to-speech/limited-access> (CNV Lite: 20-50 namuna)
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/record-custom-voice-samples> (300 / 1000 / 2000)
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/personal-voice-overview> (5–90 sek)

**Muhim raqam — "300 utterance kifoya" degan Microsoft xulosasi:**

> "in most cases you can build a reasonable custom voice with **300 utterances**. According to our tests, adding more training data in most languages doesn't necessarily improve naturalness of the voice itself (tested using the MOS score), however, with more training data that covers more word instances, you have higher possibility to reduce the ratio of dissatisfactory parts of speech"

Contextual rejim uchun alohida minimal: **30 daqiqadan ko'p audio YOKI 300 utterance**, har bir yozuv ideal holda 30 sekunddan uzun (60+ so'z lotin tillari uchun, 160+ so'z lotin bo'lmagan tillar uchun).

### 1.2 Yozish (studio) spetsifikatsiyasi — Azure e'lon qilgan aniq raqamlar

| Xususiyat | Qiymat |
|---|---|
| Fayl formati | `*.wav`, Mono |
| Sampling rate | **24 kHz** (yuqorisi ham mumkin, masalan 48 kHz / 24 bit — xizmat avtomatik 24 kHz / 16 bit PCM ga tushiradi) |
| Sample format | **16 bit, PCM** |
| Peak volume | **-3 dB dan -6 dB** gacha |
| SNR | **> 35 dB** (20 dB dan past bo'lsa generatsiya qilingan ovozda shovqin eshitiladi) |
| Sukunat | boshida va oxirida ~**100 ms** (lekin **200 ms dan uzun bo'lmasin**); so'zlar orasidagi sukunat **< -30 dB**; oxirgi so'zdan keyingi sukunat **< -60 dB** |
| Muhit shovqini | gapirishdan oldingi to'lqinda **< -70 dB** (uy sharoitida maqsad **-80 dB**) |
| Dinamik diapazon siqilishi | maksimum **4:1** |
| Utterance uzunligi | **15 sekunddan qisqa** (30 sekunddan uzun bo'lsa avtomatik rad etiladi) |

Skript tanlash mezonlari:
- gaplar **2–15 sekund**, **5–30 so'z** (lotin asosidagi tillar) yoki **4–80 so'z** (lotin bo'lmaganlar)
- **70–80% darak gaplar**, **~10% qisqa so'z/ibora** (5–7 so'z), **10–20% so'roq** (5-10% ko'tariluvchi, 5-10% tushuvchi ohang), **10–20% undov**
- takroriy gaplar yo'q; A–Z harflarining barchasi qamrab olinsin; raqam/manzil/telefon/sana formatlari kiritilsin
- **training skripti voice talent skriptidan farq qiladi**: talent uchun "50%", "$45", "123" kabi tabiiy yoziladi; training uchun **normallashtirilgan** bo'lishi kerak — "fifty percent", "forty-five dollars", "one hundred and twenty-three"
- bir qator = bir utterance; skript qatori **1000 belgidan qisqa**

Voice talent amaliyoti: ko'pchilik talent **kuniga faqat 2–3 soat** yozib bera oladi; haftada **3–4 kun** chegara, oralarida dam kuni. Studiyada odatda **10–24 utterance**lik "take" lar bilan yoziladi. Seans boshida **"match file"** (etalon yozuv) olinadi va har sahifada takrorlanadi — balandlik/tempo/ohang izchilligi uchun. Birinchi yozuvdan oldin **~5 sekund sukunat** ("room tone") yozib olinadi, xizmat shovqinni kompensatsiya qilishi uchun.

Manba: <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/record-custom-voice-samples>

### 1.3 Azure avtomatik nima qiladi — bu foydalanuvchi quradigan quvurning aynan o'zi

Bu — hujjatning markaziy qismi. Azure uchta ma'lumot turini qabul qiladi:

| Ma'lumot turi | Tavsif | Azure qo'shimcha nima qiladi |
|---|---|---|
| **Individual utterances + matching transcript** | `.zip` ichida har biri 15 sek dan qisqa `.wav` + formatlangan `.txt` | hech narsa, "Ready for fine-tuning" |
| **Long audio + transcript** | `.zip` ichida segmentlanmagan uzun audio (`.wav`/`.mp3`, **30 sekunddan uzun**, **ko'pi bilan 1000 fayl**) + barcha gapirilgan so'zlarni o'z ichiga olgan transkript `.zip` | **Segmentatsiya (Batch Transcription API orqali)** + kerak bo'lsa audio format konvertatsiyasi |
| **Audio only** | `.zip` ichida transkriptsiz audio (ko'pi bilan 1000 fayl, uzunlik cheklovsiz) | **Segmentatsiya + transkript generatsiyasi (Batch Transcription)** + format konvertatsiyasi |

Arxiv cheklovlari: **maksimum 2048 MB** zip; **500 zip** per subscription (S0); bir vaqtda **5 ta fayl** yuklash.

Segmentatsiya tafsilotlari:
- segmentlangan utterance ideal holda **5–15 sekund**
- optimal segmentatsiya uchun har **5–15 sekundda 0.5–1 sekundlik tabiiy pauza** tavsiya etiladi, afzali ibora/gap oxirida
- `.mp3` fayllar qayta ishlangandan keyin `.wav` ga aylantiriladi; `.mp3` uchun **kamida 256 kbps** bitrate
- segmentlangan utterance larga **avtomatik unikal ID** beriladi
- segmentatsiya davomida audio va transkript **custom speech** xizmatiga ham yuboriladi — tanish modelini sizning ma'lumotingizga moslashtirish uchun. "No data is retained during this process."

**Avtomatik sifat tekshiruvlari (eng qimmatli qism):**

*Avtomatik RAD ETILADI:*
| Kategoriya | Shart |
|---|---|
| Script | ID va matn orasida Tab bo'lmasa; ID raqam bo'lmasa; qator takrorlansa; 1000 belgidan uzun bo'lsa; mos audio topilmasa |
| Audio | format yaroqsiz; **sampling rate 16 kHz dan past**; **audio 30 sekunddan uzun**; mos skript yo'q |
| Mismatch | **gap darajasidagi talaffuz bahosi (pronunciation score) 70 dan past** — 0–100 shkalada. "A score below 70 normally indicates a speech error or script mismatch. A heavy accent can reduce your pronunciation score" |

*Avtomatik TO'G'RILANADI:*
- boshlanish sukunati 100 ms dan qisqa bo'lsa → 100 ms gacha cho'ziladi
- tugash sukunati 100 ms dan qisqa bo'lsa → 100 ms gacha cho'ziladi
- matn raqam/belgi/qisqartmalar uchun **avtomatik normallashtiriladi** ("Text auto normalized")
- **"Download the normalized dataset and review it"** ← datasetni qaytarib berish shu yerda ochiq aytilgan

*QO'LDA tekshirish kerak (training dan chiqarib tashlanmaydi, lekin sifatga ta'sir qiladi):*
- normallashtirilmagan matn (masalan `/` → "slash")
- **so'roq gaplar 10% dan kam**; **undov gaplar 10% dan kam**
- gap oxirida to'g'ri tinish belgisi yo'q
- sampling rate neural ovoz uchun past (24 kHz dan past → avtomatik 24 kHz ga ko'tariladi)
- umumiy balandlik **-18 dB dan past**; volume overflow (vaqt belgisi bilan ko'rsatiladi)
- boshi/oxiri sukunati "toza emas"
- "Low scored words" — so'z darajasida past baho
- **SNR 20 dB dan past** ("At least 35 dB is recommended")
- "No score available" — nutqni tanib bo'lmadi

Speech Studio / Foundry ichida har bir utterance uchun transkriptni **brauzerda tahrirlash**, yangi `.wav` yuklash, keyin **"Analyze data"** bosib qayta tahlil qilish mumkin. Qayta tahlil tugamaguncha bu training set ni tanlab bo'lmaydi.

Manbalar:
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-custom-voice-training-data>
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/professional-voice-create-training-set>

### 1.4 Training usullari va vaqti

Usullar: **Neural – HD** (LLM asosidagi, dinamik suhbat uchun), **Neural – Default**, **Neural – Multilingual** (bir tilda o'qitib, bir nechta tilda gapiradigan ovoz), **Neural – Multi Style** (ko'p uslub/emotsiya).

- Multi Style: **kamida 300 utterance** umumiy ma'lumot + har bir maxsus uslub uchun **100 utterance**; English (US) uchun **10 ta**, Chinese (Mandarin) uchun **4 ta**, Japanese uchun **5 ta** maxsus uslub
- O'rtacha training vaqti: **~10 compute hour** per model
- S0 subscription: bir vaqtda **4 ta ovoz** o'qitish mumkin
- Barcha ma'lumot **bir xil so'zlovchidan** bo'lishi shart
- Narx: "Although the total number of hours required per training method varies, the same unit price applies to each"

Manba: <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/professional-voice-train-voice>

### 1.5 Narxlar (Azure Retail Prices API, `eastus`, o'qilgan 2026-10-01)

> Azure'ning odatiy narx sahifasi (<https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/>) raqamlarni JavaScript bilan yuklaydi va `$-` ko'rsatadi. Shu sababli raqamlar rasmiy **Azure Retail Prices API** dan olindi:
> `https://prices.azure.com/api/retail/prices?$filter=productName eq 'Azure Speech' and armRegionName eq 'eastus'`

| Meter | Birlik | Narx (USD) |
|---|---|---|
| **S1 Custom Neural Training** | 1 soat (compute hour) | **$52.00** |
| **S1 Custom Neural Voice Model Hosting Unit** | 1/soat | **$4.032** (US Gov: $5.04) |
| S1 Custom Neural Realtime Characters | 1M belgi | $24.00 |
| S1 Custom Neural Long Audio Characters | 1M belgi | $100.00 |
| S1 Neural Text To Speech Characters | 1M belgi | $15.00 |
| Neural HD Text to Speech Characters | 1M belgi | $22.00 |
| Text to Speech – Personal Voice Characters | 1M belgi | $24.00 |
| Text to Speech – Personal Voice Voice Storage | 1K (ovoz) | $600.00 |
| Commitment Tier Neural TTS 80M | 1/oy | $960.00 (overage $12.00/1M) |
| Commitment Tier Neural TTS 400M | 1/oy | $4,800.00 (overage $9.75/1M) |
| Commitment Tier Neural TTS 2000M | 1/oy | $12,000.00 |
| Commitment Tier Neural TTS 4000M | 1/oy | $24,000.00 (overage $6.00/1M) |

**Amaliy hisob:** o'rtacha 10 compute hour training × $52 = **~$520** bir modelni o'qitish. Endpoint hosting $4.032/soat × 730 soat = **~$2,943/oy** — ya'ni modelni jonli ushlab turish training dan 5–6 barobar qimmat. Bu CNV ni kichik loyihalar uchun amalda yopadi.

### 1.6 Consent / razılıq verifikatsiyasi (eng qattiq rejim bozorda)

Professional voice uchun ovoz modelini o'qitishdan **oldin** voice talent ning og'zaki razılıq bayonotini yuklash **shart**. `en-US` uchun aynan matn:

```
"I [state your first and last name] am aware that recordings of my voice will be used
by [state the name of the company] to create and use a synthetic version of my voice."
```

- Barcha lokallar uchun matn: <https://github.com/Azure-Samples/Cognitive-Speech-TTS/blob/master/CustomVoice/script/verbal-statement-all-locales.txt>
- Bayonot **training ma'lumoti bilan bir xil sozlama, muhit va gapirish uslubida** yozilishi kerak
- `voiceTalentName` va `companyName` yozuvda aytilgan nom bilan **so'zma-so'z** mos kelishi kerak, **yozuv tilida**; keyin o'zgartirilmaydi
- Format: `.mp3` yoki `.wav`; REST API: `Consents_Create` / `Consents_Post`
- **Biometrik tekshirish:** "Microsoft reserves the right to use Microsoft's speaker recognition biometric identification technology on this recorded acknowledgement statement and **verify it against the training audio data to confirm that the voices are from the same speaker**." Microsoft bu faylni va ovoz modelini **saqlab qoladi**
- Mijoz GDPR Art. 4 ma'nosida biometrik ma'lumot uchun javobgar

Personal voice uchun ham: "it's required that every voice be created with explicit consent from the user. A recorded statement from the user is required."

Manbalar:
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/professional-voice-create-consent>
- <https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/speech-service/text-to-speech/limited-access>

### 1.7 Gated access — kim umuman kira oladi

> "Only customers **managed by Microsoft**, meaning those who are working directly with Microsoft account teams, are eligible for access."

- Ro'yxatdan o'tish formasi: <https://aka.ms/customneural>
- Foydalanish **faqat registratsiyada tanlangan va Microsoft tasdiqlagan use case** lar bilan cheklangan
- Microsoft ma'lumotni qayta tasdiqlashni talab qilishi mumkin
- Mijoz voice talent dan **yozma ruxsat** olganini kafolatlashi va "Disclosure for voice and avatar talent" hujjatini oldindan talentga berishi shart
- Deploy qilganda **sintetik tabiatni oshkor qilish** va foydalanuvchilar uchun feedback kanali ochish majburiy
- Personal voice ham, hatto Speech Studio dagi demo ham Limited Access

**Bu foydalanuvchi savolining asosiy javobi:** quvur mavjud, lekin O'zbekistondagi yakka dasturchi uchun **amalda yopiq** — "Microsoft account team bilan to'g'ridan-to'g'ri ishlaydigan mijoz" bo'lish shart.

### 1.8 Til qo'llovi — o'zbekcha

`language-support?tabs=tts` sahifasiga ko'ra:

| Til | Lokal | Tayyor neural ovoz | Custom voice (professional fine-tuning) |
|---|---|---|---|
| **O'zbek** | `uz-UZ` | **Yo'q** | **Yo'q** |
| Turk | `tr-TR` | Ha (AhmetNeural, EmelNeural) | ro'yxatda belgi yo'q |
| Qozoq | `kk-KZ` | Ha (AigulNeural, DauletNeural) | ro'yxatda belgi yo'q |
| Ozarbayjon | `az-AZ` | Ha (BanuNeural, BabekNeural) | ro'yxatda belgi yo'q |

Qiziq jihat: `uz-UZ` **Custom Speech** (nutqni tanish) uchun "Plain text" bilan qo'llanadi — ya'ni Azure o'zbekcha ASR ni biladi, lekin o'zbekcha TTS ni bilmaydi.

Shuningdek **"Long audio + transcript"** (ya'ni foydalanuvchiga kerak bo'lgan segmentatsiya xizmati) faqat quyidagi lokallarda ishlaydi: zh-CN, zh-HK, zh-TW, en-IN, en-GB, en-US, fr-FR, de-DE, hi-IN, it-IT, ja-JP, pt-BR, es-ES, es-MX. **O'zbek, turk, qozoq, ozarbayjon — yo'q.** Contextual rejim yana ham torroq: en, zh-CN, fr, it, de, es, pt-BR, ja, ko, cs.

Manba: <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts>

---

## 2. Google Cloud Text-to-Speech

### 2.1 Custom Voice — YANGI MIJOZ QABUL QILMAYDI

Rasmiy hujjatning birinchi qatori:

> "**Custom Voice isn't onboarding new customers.** Try out instant custom voice instead."

Manba: <https://docs.cloud.google.com/text-to-speech/custom-voice> (o'qilgan 2026-10-01)

Custom Voice "train a custom voice model using your own **studio-quality audio recordings**" degan ta'rifni saqlagan, lekin kerakli soat miqdori, studio spetsifikatsiyasi, allowlist protsedurasi va consent talablari **hozirgi sahifada e'lon qilinmagan** — **ma'lumot topilmadi**.

### 2.2 Chirp 3: Instant Custom Voice — hozirgi yo'l

| Parametr | Qiymat |
|---|---|
| Consent audio | **10 sekundga qadar**, bir kanalli |
| Reference (namuna) audio | **10 sekundga qadar**, bir kanalli, "as close to 10 seconds as possible" |
| Kodeklar | LINEAR16, PCM, MP3, M4A — har qanday sample rate |
| Audio sifati | tabiiy pauzalar va temp, minimal fon shovqini |
| Kirish | **"Access to Instant Custom Voice is restricted to allow-listed users. To request access, contact a member of the sales team."** |
| Rate limit | loyiha uchun **daqiqada 10 ta voice cloning key** |
| Tillar | 30+ til/lokal. **O'zbek yo'q.** Turkic tillardan faqat `tr-TR` (Turkish) bor |

**Consent matni majburiy va o'zgartirilmaydi** — Google o'zi beradigan jumlani o'qish shart:

```
en-US: "I am the owner of this voice and I consent to Google using this voice
        to create a synthetic voice model."
fr-FR: "Je suis le propriétaire de cette voix et j'autorise Google à utiliser
        cette voix pour créer un modèle de voix synthétique."
```

Hujjat ta'kidlaydi: "you **must** use the provided consent statement script" — o'z matningizni yozib bo'lmaydi.

Manba: <https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice>

### 2.3 Narx

Rasmiy narx sahijasi (<https://cloud.google.com/text-to-speech/pricing>) JavaScript bilan render qilinadi va WebFetch orqali jadvalni chiqarib bo'lmadi — **rasmiy manbadan tasdiqlanmadi**.

[ikkilamchi manba] Instant Custom Voice uchun **$60 per 1M belgi** ($0.00006/belgi), bepul oylik kvota yo'q — <https://diyai.io/ai-tools/audio-generation/google-cloud-text-to-speech-pricing/> va <https://costbench.com/software/ai-voice-tools/google-cloud-text-to-speech/>. **Rasmiy tasdiqlash kerak.**

Custom Voice (eski) training narxi: **ma'lumot topilmadi**.

### 2.4 Dataset qaytaradimi?

Hujjatda dataset qaytarish haqida hech narsa yo'q. 10 sekundlik namuna uchun segmentatsiya/moslash quvurи umuman kerak emas — Google bu muammoni **hal qilmaydi, balki chetlab o'tadi** (zero-shot). Foydalanuvchi uchun bu quvur emas.

---

## 3. Amazon Polly — Brand Voice

**Bu self-serve mahsulot EMAS. Bu professional xizmat shartnomasi.** AWS buni ochiq yozadi:

> "Brand Voice is a **custom engagement where you work with the Amazon Polly team** to build a Neural Text-to-Speech (NTTS) voice for the exclusive use of your organization."

Jarayon (AWS e'lon qilgan to'liq ro'yxat — uch qadam, xolos):
1. persona ni aniqlash
2. aktyor/aktrisani tanlash va nutqini yozib olish
3. modelni qurish va o'qitish → ovoz sizning AWS account ID(lar)ingizga ulanadi

Kirish: "If you are interested in building a Brand Voice using Polly, please **reach out to your AWS Account Manager** or contact us for more information."

| Savol | Javob |
|---|---|
| Kerakli soat/utterance | **ma'lumot topilmadi** — AWS e'lon qilmagan |
| Yozish spetsifikatsiyasi | **ma'lumot topilmadi** |
| Dataset tozalash quvuri tavsifi | **ma'lumot topilmadi** |
| Narx | **ma'lumot topilmadi** — "contact us" |
| Dataset qaytaradimi | **ma'lumot topilmadi** (model AWS account ga ulanadi; dataset haqida gap yo'q) |
| Ma'lum mijozlar | National Australia Bank, Bank of New Zealand |

Manbalar:
- <https://aws.amazon.com/polly/features/> (Brand Voice bo'limi)
- <https://aws.amazon.com/about-aws/whats-new/2020/02/amazon-polly-launches-brand-voice> (2020-02 ishga tushgan)
- <https://docs.aws.amazon.com/polly/latest/dg/available-voices.html>

**Til:** Polly ovozlar ro'yxatida **42 til/variant** bor. **O'zbek yo'q. Qozoq yo'q. Ozarbayjon yo'q.** Turkic tillardan faqat Turkish `tr-TR` (Filiz — standard, Burcu — neural).

---

## 4. ElevenLabs — Instant vs Professional Voice Cloning

Bu foydalanuvchi uchun eng amaliy xizmat, chunki **o'zbek tilini qo'llaydi** (quyida).

### 4.1 Instant Voice Cloning (IVC)

| Parametr | Qiymat |
|---|---|
| Minimal audio | **kamida 1 daqiqa** |
| Tavsiya | **1–2 daqiqa** toza audio (reverb, artefakt, fon shovqini yo'q) |
| Maksimum | **3 daqiqadan oshmasin** — "more than 2-3 minutes of audio will yield little improvement and can, in some cases, even be detrimental" |
| Namuna soni | cheklov yo'q — "The number of samples you use doesn't matter; it is the **total combined length** (total runtime) that is the important part" |
| Tier | Starter ($6/oy) va yuqorisi |
| Consent | "confirm that you have the right and consent to clone the voice" — oddiy checkbox |
| Tezlik | deyarli bir zumda |

Manba: <https://elevenlabs.io/docs/product-guides/voices/voice-cloning/instant-voice-cloning>

### 4.2 Professional Voice Cloning (PVC)

| Parametr | Qiymat |
|---|---|
| **Minimal audio** | **30 daqiqa** |
| **Tavsiya** | **2–3 soat** — "for the optimal result and the most accurate clone, we recommend closer to 2-3 hours of audio" |
| Fine-tune vaqti | **3–6 soat**, navbatga qarab **24 soatgacha** cho'zilishi mumkin |
| Tier | **Creator va yuqorisi** |
| RMS balandlik | **-23 dB dan -18 dB** orasida izchil |
| Audio talab | bitta so'zlovchi, fon shovqini/musiqa yo'q, akustik ishlov berilgan xona tavsiya etiladi |
| Tillar | "all languages available in the **Eleven v4** model family — **90+ languages**" |

**Verifikatsiya (voice captcha) — ElevenLabs ning eng qattiq cheklovi:**

> "For now, we only allow you to clone **your own voice**. You will be asked to go through a verification process before submitting your fine-tuning request."
>
> FAQ: "**No. You can only create a Professional Voice Clone of your own voice. Even with their consent, you cannot clone someone else's voice.**"

- Mikrofon orqali verifikatsiya qatorlarini o'qib yozib berasiz
- "use the same or similar equipment used to record the samples and in a tone and delivery that is similar"
- Barcha urinish muvaffaqiyatsiz bo'lsa: **24 soat kutish** kerak, yoki support ga murojaat. Aniq urinish soni hujjatda ko'rsatilmagan — **ma'lumot topilmadi**

**Dataset qaytaradimi?** YO'Q, ochiq aytilgan:

> "**You cannot download or export your voice clones as standalone files. Voice clones stay in your ElevenLabs account.**"

Qayta ishlangan (segmentlangan/tozalangan) training ma'lumotini qaytarish haqida hujjatda hech narsa yo'q — ya'ni amalda yo'q.

Manba: <https://elevenlabs.io/docs/product-guides/voices/voice-cloning/professional-voice-cloning>

### 4.3 Narxlar (elevenlabs.io/pricing, o'qilgan 2026-10-01)

| Tier | Narx/oy | Kreditlar/oy | PVC slot |
|---|---|---|---|
| Free | $0 | 10,000 | 0 |
| Starter | $6 | 30,000 | 0 (faqat IVC) |
| Creator | $11–$22 (pastga qarang) | 121,000 | 1 |
| Pro | $99 | 600,000 | 5 (narx sahifasi) / 1 (docs) |
| Scale | $299 | 1,800,000 | 3 |
| Business | $990 | 6,000,000 | 10 |
| Enterprise | custom | custom | custom |

**MANBALAR ZIDDIYATI (ikkitasini ham beraman):**
1. *Creator narxi:* narx sahifasida "$22 First month 50% off" va "$11 per month" ikkalasi ko'rinadi — ya'ni to'liq narx $22/oy, birinchi oy $11 bo'lishi ehtimoli katta, lekin sahifa matnidan aniq ajratib bo'lmadi.
2. *PVC slotlari:* narx sahifasi Pro = 5 slot deydi; hujjat sahifasi "Creator, Pro, legacy Scale: **1** PVC slot; Scale, legacy Business: 3; Business: 10" deydi. Ikkalasi ham keltirilgan.

### 4.4 Til — O'ZBEK BOR

| Model | Tillar | O'zbek | Turk | Qozoq | Ozarbayjon |
|---|---|---|---|---|---|
| **Eleven v4** | 90+ | **HA — "Uzbek (uzb)"** | ha (tur) | ha (kaz) | ha (aze) |
| Eleven v3 | 70+ | **yo'q** | ha | ha | ha |
| Multilingual v2 | 29 | yo'q | ro'yxat qisqa, tasdiqlash kerak | yo'q | yo'q |
| Flash v2.5 | 32 | yo'q | tasdiqlash kerak | yo'q | yo'q |

PVC "all languages available in the Eleven v4 model family" ni qo'llashi sababli — **ElevenLabs PVC o'zbek tili uchun ishlaydi**. Bu, bozorda topilgan **yagona** o'zbekchani qo'llovchi professional (30 daqiqa – 3 soat ma'lumotli) klonlash yo'li.

Manba: <https://elevenlabs.io/docs/models>

---

## 5. Qolgan xizmatlar

### 5.1 Resemble AI

| Parametr | Qiymat |
|---|---|
| Minimal audio | **10 sekund – 3 daqiqa** |
| Usul A (Dataset File) | bitta `.wav`, **≥10 sekund** |
| Usul B (Individual Recordings) | **3+ yozuv, ≈10 sekund jami** |
| Training vaqti | **1 daqiqadan kam** |
| Tier | docs: "**Business plan or higher**" |
| Professional/studio tier | hujjatda rapid vs professional ajratilmagan — **ma'lumot topilmadi** |
| Consent jarayoni | hujjatda ko'rsatilmagan — **ma'lumot topilmadi** |
| Dataset qaytaradimi | **ma'lumot topilmadi** |

Narxlar (resemble.ai/pricing, o'qilgan 2026-10-01): **Flex $0/oy** (pay-as-you-go), **Team $350/oy** ($280 yillik), **Business $1,000/oy** ($800 yillik), **Enterprise** custom. Resemble Identity search $0.00050/call. Qaysi tier klonlashni ochishi narx sahifasida aniq yozilmagan (hujjat "Business plan or higher" deydi).

Manbalar: <https://docs.resemble.ai/voice-creation/voices/clone-overview>, <https://www.resemble.ai/pricing/>

### 5.2 Respeecher

- Asosiy mahsulot: **speech-to-speech ovoz konvertatsiyasi** (klassik TTS emas) + **Voice Marketplace** (40+ AI ovoz) + yangi real-time TTS API
- Model: gibrid — marketplace self-serve (bepul test bor), speech-to-speech va custom loyihalar uchun "Contact us"
- Narx: TTS API uchun **"$2/hour, cancel anytime"**; custom loyihalar uchun narx e'lon qilinmagan
- Kerakli ma'lumot miqdori: **ma'lumot topilmadi** — sayt aniq soat/daqiqa bermaydi
- Consent: "Strict consent mechanisms", "Transparent voice origin tracking" deb e'lon qiladi, lekin texnik protsedura hujjatlanmagan
- Dataset qaytaradimi: **ma'lumot topilmadi**

Manba: <https://www.respeecher.com/>

### 5.3 Play.ht / PlayAI

Rasmiy help sahifalari (`help.play.ai`, `help.play.ht`) TLS/DNS xatolari bilan ochilmadi — quyidagilar **kompaniyaning o'z blogi va qidiruv natijasidan** [ikkilamchi manba]:

- **Instant cloning:** **~30 sekund** audio, natija deyarli bir zumda
- **High-fidelity cloning:** *ziddiyatli raqamlar:*
  - play.ht blogi: "**kamida 10 daqiqa, ideal 1–2 soat**; nozik aksent uchun **6 soatgacha**"
  - PlayAI uchun: "eng yuqori sifatli klonlar **20 daqiqa** namuna talab qiladi, eng yaxshi natija uchun **30 daqiqa** tavsiya etiladi"
- Qayta ishlash vaqti: **20 daqiqadan bir necha soatgacha**, yuklangan audio uzunligiga qarab
- Narx / tier: `play.ht` va `www.play.ai` domenlari fetch da javob bermadi — **ma'lumot topilmadi**
- Dataset qaytaradimi: **ma'lumot topilmadi**

Manbalar [ikkilamchi]: <https://play.ht/blog/ai-voice-cloning-tips/>, <https://techcrunch.com/2024/11/25/playai-clones-voices-on-command/>

### 5.4 Cartesia

| Parametr | Qiymat |
|---|---|
| **Instant Voice Clone** — minimal | **10 sekund** — "10 seconds of audio is enough to get started" |
| Instant — maksimum | **60 sekundgacha** (Sonic 3.6+ modellari; eski modellar faqat birinchi 10 sekundni ishlatadi) |
| Fayl cheklovi | **16 MB dan katta bo'lmasin** |
| Audio talab | bitta so'zlovchi, fon shovqini yo'q |
| **Pro Voice Clone** | **"30 daqiqa yoki ko'proq"** bir xil so'zlovchi audiosi — murakkab ovozlarning o'ziga xos sifatlarini saqlash uchun |
| Til | avval so'zlovchining ona tilida klonlang; qo'shimcha tillar "Add Voice Accents API" orqali |
| Narx / tier | hujjatning qolgan qismi auth talab qiladi — **ma'lumot topilmadi** |
| Consent | **ma'lumot topilmadi** |
| Dataset qaytaradimi | **ma'lumot topilmadi** |

Manba: <https://docs.cartesia.ai/build-with-cartesia/capability-guides/clone-voices>

### 5.5 Descript Overdub

[ikkilamchi manba — rasmiy help sahifasi topilmadi, qidiruv natijasi Descript hujjat/terms ga asoslanadi]

- **Minimal: 10 daqiqa** training audio
- **Tavsiya: 30–90 daqiqa** production sifati uchun
- **Consent:** training jarayonining bir qismi sifatida foydalanuvchi **shaxsini tasdiqlovchi va Overdub Voice yaratishga razılıq beruvchi skriptni o'qishi shart**. "Training data which does not include this consent statement **cannot be used** to create an Overdub Voice"
- Narx/tier: **ma'lumot topilmadi**
- Dataset qaytaradimi: **ma'lumot topilmadi**

Manba [ikkilamchi]: <https://www.descript.com/terms>, qidiruv natijalari

### 5.6 WellSaid Labs

- 280+ tayyor ovoz, 50+ til/aksent, barchasi **professional voice aktyorlar bilan litsenziyalangan yozuvlardan**
- "Studio-quality AI voice in seconds, made in partnership with professional voice actors"
- Custom voice avatar uchun kerakli ma'lumot miqdori, studio talablari, self-serve/sales modeli, narx: **barchasi ma'lumot topilmadi** — "Book a demo"
- Bepul trial bor (karta kerak emas)
- Dataset qaytaradimi: **ma'lumot topilmadi**

Manba: <https://www.wellsaid.io/>

### 5.7 Murf AI — hujjatlangan ikki darajali model

Murf ning rasmiy API hujjati eng aniq ajratishni beradi:

| | Instant Voice Cloning | Professional Voice Cloning |
|---|---|---|
| Audio | **bitta qisqa namuna, maks. 30 sekund, 40 MB gacha** | **90 daqiqagacha studio sifatli yozuv** |
| Format | WAV, MP3, FLAC, ALAW, ULAW; **min. 24 kHz** | — |
| Talab | bitta so'zlovchidan aniq, uzluksiz nutq, minimal fon shovqini | — |
| Turnaround | bir zumda | **1–4 hafta, Murf jamoasi boshqaradi** |
| Tier | **faqat Enterprise plan**, workspace uchun sales yoqadi, **faqat API** (Studio da yo'q), Falcon 2 modeli | **faqat Enterprise**, Studio + API, Falcon 2 va Gen2 |
| Consent | "Only clone a voice you own or have **clear, documented consent** to clone" | shu |
| Til | klonlar "multilingual out of the box" — bir tilda yozilgan namuna Falcon 2 qo'llagan har qanday lokalda gapiradi; sayt "20+ til" deydi, **o'zbek ro'yxatda yo'q** |

Narx: e'lon qilinmagan — sales bilan bog'lanish kerak. [ikkilamchi manba] uchinchi tomon tahlili Enterprise shartnomasini **$1,000–$5,000+/yil** deb baholaydi (<https://www.layer3labs.io/guides/is-murf-ai-worth-it>) — **rasmiy tasdiqlanmagan**.

Dataset qaytaradimi: **ma'lumot topilmadi**.

Manbalar: <https://murf.ai/api/docs/voices-styles/voice-cloning>, <https://murf.ai/voice-cloning>

### 5.8 Speechify

Rasmiy API hujjatiga ko'ra:

| Parametr | Qiymat |
|---|---|
| Audio | **10–30 sekund** (1 daqiqadan kam), **5 MB dan kichik** |
| Audio talab | aniq nutq, minimal fon shovqini, tinch xona, tabiiy va izchil temp, uzun pauza/shivirlash yo'q |
| **Consent — ikki qadamli** | 1) Speechify **challenge phrase** (sinov jumlasi) beradi, so'zlovchi uni ovoz chiqarib o'qib yozadi — bu razılıq dalili; 2) bu consent yozuvi namuna bilan birga yuboriladi. Consent yozuvi berilgan jumlaga mos kelishi va **namuna bilan bir xil shaxsdan** bo'lishi shart |
| Kirish | shu funksiyani o'z ichiga olgan plan dagi workspace + kamida bitta API key |
| Til | `simba-3.2` klonlari **faqat ingliz**; `simba-3.0` — ingliz, nemis, ispan, fransuz, italyan, portugal. **O'zbek yo'q** |
| Professional tier | **ma'lumot topilmadi** |
| Dataset qaytaradimi | **ma'lumot topilmadi** |

Manba: <https://docs.speechify.ai/docs/features/voice-cloning>

---

## 6. Ovoz marketplace'lari va ma'lumot yetkazib beruvchilar — tayyor TTS dataset sotadimi?

Savol: kimdir **tayyorlangan, TTS-ga tayyor datasetni** (xom audio emas) sotadimi, soatiga narx bilan?

### 6.1 Defined.ai (eski DefinedCrowd) — yagona OCHIQ narxli

Marketplace faol; ISO 42001 sertifikati olgan; 1.6M+ global ekspert, 150+ mamlakat; **4M+ soat audio**, 175+ domen, "500+ Languages and locales".

**Data Access Plan (DAP) — e'lon qilingan 2024-09-26, ochiq narxlar:**

| Plan | Soat | Narx (EUR) | Chegirma | **Soatiga** |
|---|---|---|---|---|
| Standard | 315 | €50,000 | 30% | **~€159/soat** |
| Premium | 625 | €100,000 | 40% | **~€160/soat** |
| Ultimate | 1,565 | €250,000 | 50% | **~€160/soat** |
| Elite | 3,125 | €500,000 | 55% | **~€160/soat** |

Manba: <https://defined.ai/press-room/defined-ai-launches-data-access-plan-dap-allowing-access-to-speech-datasets-at-affordable-prices>

TTS mahsulotlari: katalogda **"on-device scripted monologues"** ASR va **TTS** uchun ko'rsatilgan; misol mahsulot: **"English Female Voice Actor"** dataseti TTS ni qo'llaydi. Ya'ni ha, scripted monologue korpuslari TTS uchun sotiladi.

[ikkilamchi manba] katalog datasetlari **$60–95/soat** oralig'ida, 60 til, har bir datasetda 500–2,000 soat va 50–200 ona tili so'zlovchisi — <https://www.speechdata.ai/guides/how-to-buy-speech-data>. **Rasmiy tasdiqlanmagan.**

**O'zbek:** katalogda alohida ko'rsatilmagan — **ma'lumot topilmadi**.

Manba: <https://defined.ai/dataset-type/audio>

### 6.2 Appen

Off-the-shelf datasets sahifasida **"Speech & Audio"** kategoriyasi bor: "Read, conversational and call centre speech across a wide range of languages and accents". Shuningdek **"Pronunciation & POS Dictionaries"** va Multimodal AI ostida **"expressive TTS annotation"** eslatilgan. "500+ global locales".

- Soat miqdori: **e'lon qilinmagan**
- Narx: **e'lon qilinmagan** — "Talk to an expert"
- O'zbek: **eslatilmagan**

Manba: <https://appen.com/off-the-shelf-datasets/>

### 6.3 Shaip, Sama, Scale AI, Summa Linguae, Daily Transcription

| Yetkazib beruvchi | Natija |
|---|---|
| **Shaip** | sinab ko'rilgan URL lar (`/offerings/audio-speech-data-collection-and-licensing/`, `/offerings/audio-speech-datasets/`, `/offerings/off-the-shelf-datasets/`) **404** qaytardi — **ma'lumot topilmadi** |
| **Sama** | nutq korpusi mahsuloti e'lon qilinmagan (annotatsiya/labeling xizmati) — **ma'lumot topilmadi** |
| **Scale AI** | TTS korpusi mahsuloti e'lon qilinmagan — **ma'lumot topilmadi** |
| **Summa Linguae** | domen `datamundi.ai` ga ko'chgan; `403 Forbidden` — **ma'lumot topilmadi** |
| **Daily Transcription** | narx sahifasi **404** — **ma'lumot topilmadi** |
| **Voices.com** | voice-actor marketplace; "AI training uchun ovoz ma'lumoti" bo'yicha blog yuritadi, lekin mahsulot katalogi/narxi tasdiqlanmadi — **ma'lumot topilmadi** |

### 6.4 Bu bo'limning xulosasi

**Hech biri foydalanuvchi so'ragan xizmatni sotmaydi.** Ular **o'zlari yig'gan ma'lumotni** sotadi (soatiga ~€160, Defined.ai), yoki **buyurtma bo'yicha yangi ma'lumot yig'adi**. "Mening uzun yozuvim + skriptimni bering, segmentlangan datasetni qaytaring" degan xizmat — **yo'q**. Barchasi "contact sales" modeli, minimal shartnoma o'n minglab yevro.

---

## 7. MARKAZIY SAVOL: uzun yozuv + skript bersa, segmentlangan datasetni qaytaradimi?

Bu foydalanuvchi savolining yuragi. Aniq javob:

### 7.1 FAQAT BITTA xizmat buni qiladi: Azure

Azure `Long audio + transcript` yoki `Audio only` turini qabul qiladi va:
1. Batch Transcription API bilan **segmentlaydi**
2. Skriptga **moslaydi** (yoki ASR bilan transkript **yaratadi**)
3. Har bir utterance ga **unikal ID** beradi
4. **Talaffuz bahosi** (0–100), **SNR**, **volume**, **sukunat** tekshiradi
5. Sukunatlarni **avtomatik to'g'rilaydi**, matnni **normallashtiradi**
6. `.mp3` → `.wav`, sample rate ni 24 kHz ga keltiradi
7. **Natijani yuklab olish mumkin** — hujjatdan to'g'ridan-to'g'ri:

> "After the segmentation is done, only the segmented utterances and their mapping transcripts are stored **for your downloading** and fine-tuning."
>
> "You can check the segmented utterances and the matching transcripts **by downloading the dataset**. Unique IDs are assigned to the segmented utterances automatically."
>
> "Silence auto fixed ... **Download the normalized dataset and review it.**"

### 7.2 Lekin bu "xizmat" emas, yon mahsul — va to'rtta devor bor

1. **Gated:** Limited Access, faqat "customers managed by Microsoft, meaning those who are working directly with Microsoft account teams". O'zbekistondagi yakka dasturchi bu shartni qondirmaydi.
2. **Til:** segmentatsiya faqat 14 lokalda ishlaydi — zh-CN/HK/TW, en-IN/GB/US, fr-FR, de-DE, hi-IN, it-IT, ja-JP, pt-BR, es-ES, es-MX. **O'zbek yo'q.** Uzbek uchun `uz-UZ` da tayyor neural ovoz ham, custom voice ham yo'q.
3. **Bog'langan sotuv:** bu quvur mustaqil mahsulot emas. Siz dataset uchun kelmaysiz — siz ovoz modeli uchun kelasiz, dataset esa yo'lda qoladigan artefakt. Alohida "dataset tayyorlash" SKU si **yo'q**; training $52/compute hour, hosting $4.032/soat.
4. **Lock-in:** modelni Azure endpoint sifatida ushlab turish kerak — oyiga ~$2,943.

### 7.3 Qolgan hammasi — qora quti

| Xizmat | Datasetni qaytaradimi |
|---|---|
| Google Instant Custom Voice | yo'q (10 sek namuna — quvur umuman yo'q) |
| Google Custom Voice | yangi mijoz qabul qilmaydi |
| Amazon Polly Brand Voice | **ma'lumot topilmadi** (model AWS account ID ga ulanadi) |
| **ElevenLabs** | **YO'Q — ochiq rad etilgan:** "You cannot download or export your voice clones as standalone files. Voice clones stay in your ElevenLabs account." |
| Resemble, Respeecher, Play.ht, Cartesia, Descript, WellSaid, Murf, Speechify | hech biri dataset qaytarishni e'lon qilmaydi — **ma'lumot topilmadi**; amaliyotda yo'q |
| Defined.ai / Appen va boshqalar | ular **o'z** datasetini sotadi, sizning audiongizni qayta ishlamaydi |

### 7.4 Nega shunday? (xizmatlarning o'z mantiqi)

Bu bo'limning faktlaridan ko'rinadigan biznes mantiq: dataset — bu **moat**. Agar xizmat tozalangan, segmentlangan, moslangan datasetni qaytarsa, mijoz uni olib ketib istalgan open-source model (VITS, XTTS, F5-TTS...) ni o'qitadi va xizmatni tashlab ketadi. Shu sababli Azure dan tashqari hamma faqat **inference huquqini** sotadi, ma'lumotni emas. Azure esa buni faqat shuning uchun ochiq qilgan — uning mijozi allaqachon enterprise shartnomada va ma'lumot sifatini o'zi tekshirishi kerak (talaffuz bahosi past utterance larni qayta yozish uchun).

Ikkinchi sabab — **verifikatsiya majburiyati**. ElevenLabs PVC faqat **o'z ovozingizni** klonlashga ruxsat beradi (razılıq bo'lsa ham boshqasini yo'q). Azure biometrik speaker recognition bilan consent yozuvini training audio ga qarshi tekshiradi. Google majburiy consent jumlasi beradi. Speechify challenge phrase chiqaradi. Ya'ni "ovoz bering, dataset olib ketasiz" modeli — identifikatsiya zanjirini uzadi va hech bir yirik provayder bunga rozi bo'lmaydi.

---

## 8. O'zbek / Turkic til qo'llovi — barcha xizmatlar kesimida

| Xizmat | O'zbek (uz) | Turk (tr) | Qozoq (kk) | Ozar (az) | Izoh |
|---|---|---|---|---|---|
| Azure prebuilt TTS | **yo'q** | ha | ha | ha | `uz-UZ` da hech qanday ovoz yo'q |
| Azure custom voice | **yo'q** | ro'yxatda belgi yo'q | yo'q | yo'q | language-support sahifasiga ko'ra |
| Azure long-audio segmentatsiya | **yo'q** | yo'q | yo'q | yo'q | faqat 14 lokal |
| Azure Custom Speech (ASR) | **ha** (Plain text) | ha (to'liq) | ha (Plain text) | — | ASR bor, TTS yo'q |
| Google Instant Custom Voice | **yo'q** | ha (tr-TR) | yo'q | yo'q | 30+ til ro'yxati |
| Amazon Polly | **yo'q** | ha (Filiz, Burcu) | **yo'q** | **yo'q** | 42 til/variant |
| **ElevenLabs Eleven v4 + PVC** | **HA (`uzb`)** | ha | ha | ha | **yagona professional yo'l** |
| ElevenLabs Eleven v3 | yo'q | ha | ha | ha | 70+ til |
| Murf | **yo'q** | ro'yxatda yo'q | yo'q | yo'q | "20+ til" |
| Speechify | **yo'q** | yo'q | yo'q | yo'q | simba-3.2 faqat en; simba-3.0 — 6 til |
| Cartesia | tasdiqlanmadi | tasdiqlanmadi | — | — | hujjat auth ortida |
| Resemble / Respeecher / Play.ht / WellSaid / Descript | **ma'lumot topilmadi** | — | — | — | til ro'yxatlari tasdiqlanmadi |

**Xulosa:** o'zbek tili uchun sintez qiladigan **yagona tasdiqlangan tijorat professional klonlash yo'li — ElevenLabs PVC (Eleven v4)**. U 30 daqiqa – 3 soat audio oladi, lekin: (a) faqat **o'z ovozingizni** klonlay olasiz, (b) datasetni **qaytarmaydi**, (c) model ElevenLabs akkauntingizda qoladi.

---

## 9. ASOSIY JADVAL

| Xizmat | Klonlash uchun min. ma'lumot | Pro tier ma'lumoti | Narx (2026-10-01) | Consent jarayoni | Datasetni qaytaradimi? | O'zbek? | Link |
|---|---|---|---|---|---|---|---|
| **Azure CNV Lite** | 20–50 namuna (Microsoft skripti) | — | $52/compute h training; $4.032/h hosting | og'zaki bayonot + biometrik tekshirish | — (demo) | yo'q | [docs](https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/speech-service/text-to-speech/limited-access) |
| **Azure Professional voice** | **300 utterance** | 1000–2000 utterance; contextual: >30 min / 300 utt | $52/compute h (~10 h ≈ $520) + $4.032/h hosting (~$2,943/oy) + $24/1M belgi | majburiy og'zaki bayonot; Microsoft speaker recognition bilan training audio ga qarshi tekshiradi; Limited Access, faqat Microsoft-managed mijozlar | **HA — segmentlangan + moslangan datasetni yuklab olish mumkin** (lekin 14 lokal, uz yo'q) | **yo'q** | [docs](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-custom-voice-training-data) |
| **Azure Personal voice** | **5–90 sekund** | — | $24/1M belgi + $600/1K voice storage | majburiy yozilgan razılıq; Limited Access | yo'q | yo'q (91 til ro'yxati) | [docs](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/personal-voice-overview) |
| **Google Custom Voice** | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | — | **yangi mijoz qabul qilmaydi** | yo'q | [docs](https://docs.cloud.google.com/text-to-speech/custom-voice) |
| **Google Instant Custom Voice** | **10 sekund** (+10 sek consent) | yo'q | [ikkilamchi] $60/1M belgi | **majburiy Google consent jumlasi**, so'zma-so'z; allowlist — sales | yo'q | **yo'q** | [docs](https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice) |
| **Amazon Polly Brand Voice** | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | **yo'q** | [features](https://aws.amazon.com/polly/features/) |
| **ElevenLabs IVC** | **1 daqiqa** (tavsiya 1–2, maks 3) | — | Starter $6/oy+ | checkbox razılıq | yo'q | **ha (v4)** | [docs](https://elevenlabs.io/docs/product-guides/voices/voice-cloning/instant-voice-cloning) |
| **ElevenLabs PVC** | **30 daqiqa** | **2–3 soat** (tavsiya); fine-tune 3–6 h, 24 h gacha | Creator $11–22/oy (1 slot) … Business $990/oy (10 slot) | **voice captcha: mikrofonda verifikatsiya qatorlari; faqat O'Z ovozingiz**; muvaffaqiyatsiz → 24 h kutish | **YO'Q** — "cannot download or export your voice clones" | **HA (Uzbek `uzb`, Eleven v4)** | [docs](https://elevenlabs.io/docs/product-guides/voices/voice-cloning/professional-voice-cloning) |
| **Resemble AI** | **10 sek – 3 min**; 3+ yozuv ≈10 sek | ajratilmagan — ma'lumot topilmadi | Flex $0 PAYG; Team $350/oy; Business $1,000/oy; docs: "Business plan or higher" | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | [docs](https://docs.resemble.ai/voice-creation/voices/clone-overview) |
| **Respeecher** | ma'lumot topilmadi | ma'lumot topilmadi | TTS API **$2/soat**; custom — contact sales | "strict consent mechanisms" (protsedura hujjatlanmagan) | ma'lumot topilmadi | ma'lumot topilmadi | [site](https://www.respeecher.com/) |
| **Play.ht / PlayAI** | ~30 sekund (instant) | [ziddiyat] 10 min–2 soat (6 soatgacha) **yoki** 20–30 daqiqa | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | [blog](https://play.ht/blog/ai-voice-cloning-tips/) |
| **Cartesia** | **10 sek** (60 sek gacha, ≤16 MB) | **Pro Voice Clone: 30+ daqiqa** | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | tasdiqlanmadi | [docs](https://docs.cartesia.ai/build-with-cartesia/capability-guides/clone-voices) |
| **Descript Overdub** | **10 daqiqa** | **30–90 daqiqa** | ma'lumot topilmadi | **razılıq skriptini o'qish SHART** — bo'lmasa ma'lumot ishlatilmaydi | ma'lumot topilmadi | ma'lumot topilmadi | [terms](https://www.descript.com/terms) |
| **WellSaid Labs** | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi (book a demo) | voice actor litsenziyasi | ma'lumot topilmadi | ma'lumot topilmadi | [site](https://www.wellsaid.io/) |
| **Murf AI** | **30 sekund** (≤40 MB, ≥24 kHz) | **90 daqiqagacha studio**; **1–4 hafta**, Murf jamoasi boshqaradi | **faqat Enterprise**, sales yoqadi; [ikkilamchi] $1,000–5,000+/yil | "voice you own or clear, documented consent" | ma'lumot topilmadi | **yo'q** | [docs](https://murf.ai/api/docs/voices-styles/voice-cloning) |
| **Speechify** | **10–30 sekund** (≤5 MB) | ma'lumot topilmadi | plan + API key kerak; narx ma'lumot topilmadi | **ikki qadamli: Speechify challenge phrase beradi → o'qib yozasiz → namuna bilan birga yuboriladi**, bir xil shaxs tekshiriladi | ma'lumot topilmadi | **yo'q** | [docs](https://docs.speechify.ai/docs/features/voice-cloning) |
| **Defined.ai** (ma'lumot sotuvchi) | — (o'z datasetini sotadi) | scripted monologue TTS korpuslari bor | **DAP: 315 h/€50,000 … 3,125 h/€500,000 ≈ €160/soat**; [ikkilamchi] katalog $60–95/soat | — | sotadi, lekin **o'z** ma'lumotini | ma'lumot topilmadi | [DAP](https://defined.ai/press-room/defined-ai-launches-data-access-plan-dap-allowing-access-to-speech-datasets-at-affordable-prices) |
| **Appen** | — | "Speech & Audio", "expressive TTS annotation" | e'lon qilinmagan | — | o'z ma'lumotini | eslatilmagan | [datasets](https://appen.com/off-the-shelf-datasets/) |
| **Shaip / Sama / Scale / Summa Linguae / Daily Transcription** | — | — | **ma'lumot topilmadi** (URL lar 404/403) | — | — | ma'lumot topilmadi | — |

---

## 10. Self-serve vs "contact sales" — aniq ajratish

**To'liq self-serve (karta bilan bugun boshlash mumkin):**
- ElevenLabs IVC (Starter $6/oy) va PVC (Creator tier)
- Resemble AI (Flex $0 PAYG — lekin docs klonlash uchun Business plan deydi; ziddiyat)
- Cartesia instant clone (API)
- Speechify (plan + API key)
- Descript Overdub

**Ro'yxatga olish / tasdiq talab qiladi (lekin self-serve API):**
- Azure professional voice, personal voice, CNV Lite — Limited Access forma + Microsoft-managed mijoz sharti
- Google Instant Custom Voice — allowlist, sales orqali

**Faqat professional xizmat shartnomasi (self-serve EMAS):**
- **Amazon Polly Brand Voice** — "custom engagement... work with the Amazon Polly team"; AWS Account Manager
- **Murf** Instant VA Professional cloning — ikkalasi ham faqat Enterprise, workspace sales tomonidan yoqiladi
- **Murf Professional** — 1–4 hafta, Murf jamoasi boshqaradi
- **WellSaid** custom voice — book a demo
- **Respeecher** speech-to-speech va custom loyihalar
- **Google Custom Voice** (eski) — yopilgan
- Barcha ma'lumot yetkazib beruvchilar (Defined.ai, Appen, Shaip, Sama, Scale, Summa Linguae)

---

## 11. Foydalanuvchi loyihasi uchun to'g'ridan-to'g'ri xulosalar

1. **Quvur texnik jihatdan isbotlangan.** Azure uni e'lon qilgan va ishlatadi: uzun audio + skript → Batch Transcription segmentatsiya → moslash → talaffuz bahosi (0–100, 70 kesish) → SNR (20 dB kesish, 35 dB maqsad) → sukunat auto-fix (100 ms) → matn normallashtirish → yuklab olinadigan dataset. Foydalanuvchi aynan shu arxitekturani qurishi mumkin, chunki uni tasavvur qilgan emas — u ishlab turgan mahsulot.

2. **"Nega hech kim qurmagan" savolining javobi: qurgan, lekin mahsulot qilib sotmagan.** Hech bir provayder datasetni **o'z-o'zidan mahsulot** sifatida sotmaydi — u faqat ovoz modeli sotuvining ichida yashiringan. Alohida "dataset tayyorlash" SKU si butun bozorda **mavjud emas**.

3. **O'zbek tili uchun bo'shliq to'liq.** Azure segmentatsiyasi 14 lokal, uz yo'q. Google 30+ til, uz yo'q. Polly 42 til, uz yo'q. Murf, Speechify — yo'q. Faqat ElevenLabs v4 o'zbekni qo'llaydi, lekin dataset qaytarmaydi va faqat o'z ovozingizni klonlashga ruxsat beradi.

4. **Benchmark raqamlar (foydalanuvchi quvuri uchun maqsad):** 300 utterance minimal; 5–15 sekund utterance; 24 kHz / 16-bit PCM mono; SNR >35 dB; peak -3…-6 dB; 100 ms boshlanish/tugash sukunati; 70–80% darak / 10% qisqa / 10–20% so'roq / 10–20% undov; talaffuz bahosi kesimi 70; matn normallashtirish majburiy.

5. **Narx taqqoslash nuqtasi:** tayyor nutq ma'lumoti bozorda **~€160/soat** (Defined.ai DAP, ochiq narx). Azure da 10 soatlik training ~$520 + oyiga ~$2,943 hosting. Ya'ni agar foydalanuvchi o'zbekcha uchun 10 soatlik tayyor dataset ishlab chiqaradigan avtomatik quvur qursa, u yaratgan artefaktning bozor qiymati **~€1,600** — va bu o'zbek tili uchun hozir sotib olinadigan joy yo'q.

6. **Consent me'morchiligi — majburiy dars.** Barcha yirik provayder razılıqni **audio dalil** sifatida oladi (Azure: nomli bayonot + biometrik moslik; Google: majburiy jumla; Speechify: challenge phrase; Descript: training skriptining bir qismi; ElevenLabs: mikrofon captcha + faqat o'z ovozi). Agar foydalanuvchi quvurini xizmat qilib sotsa, bu qatlamsiz ishlamaydi.

---

## 12. Barcha manbalar

**Microsoft Azure**
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/custom-neural-voice>
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/record-custom-voice-samples>
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-custom-voice-training-data>
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/professional-voice-create-training-set>
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/professional-voice-create-consent>
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/professional-voice-train-voice>
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/personal-voice-overview>
- <https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts>
- <https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/speech-service/text-to-speech/limited-access>
- <https://aka.ms/customneural> (Limited Access forma)
- <https://github.com/Azure-Samples/Cognitive-Speech-TTS/blob/master/CustomVoice/script/verbal-statement-all-locales.txt>
- <https://github.com/Azure-Samples/Cognitive-Speech-TTS/tree/master/CustomVoice/script> (namuna skriptlar)
- <https://prices.azure.com/api/retail/prices?$filter=productName eq 'Azure Speech' and armRegionName eq 'eastus'> (narxlar, 2026-10-01)

**Google Cloud**
- <https://docs.cloud.google.com/text-to-speech/custom-voice>
- <https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice>
- <https://cloud.google.com/text-to-speech/pricing> (JS render — jadval chiqarib bo'lmadi)

**Amazon**
- <https://aws.amazon.com/polly/features/>
- <https://aws.amazon.com/about-aws/whats-new/2020/02/amazon-polly-launches-brand-voice>
- <https://docs.aws.amazon.com/polly/latest/dg/available-voices.html>

**ElevenLabs**
- <https://elevenlabs.io/docs/product-guides/voices/voice-cloning/professional-voice-cloning>
- <https://elevenlabs.io/docs/product-guides/voices/voice-cloning/instant-voice-cloning>
- <https://elevenlabs.io/docs/models>
- <https://elevenlabs.io/pricing>

**Boshqalar**
- <https://docs.resemble.ai/voice-creation/voices/clone-overview>
- <https://www.resemble.ai/pricing/>
- <https://www.respeecher.com/>
- <https://play.ht/blog/ai-voice-cloning-tips/> [ikkilamchi]
- <https://techcrunch.com/2024/11/25/playai-clones-voices-on-command/> [ikkilamchi]
- <https://docs.cartesia.ai/build-with-cartesia/capability-guides/clone-voices>
- <https://www.descript.com/terms> [ikkilamchi]
- <https://www.wellsaid.io/>
- <https://murf.ai/api/docs/voices-styles/voice-cloning>
- <https://murf.ai/voice-cloning>
- <https://docs.speechify.ai/docs/features/voice-cloning>

**Ma'lumot yetkazib beruvchilar**
- <https://defined.ai/press-room/defined-ai-launches-data-access-plan-dap-allowing-access-to-speech-datasets-at-affordable-prices>
- <https://defined.ai/dataset-type/audio>
- <https://www.defined.ai/>
- <https://appen.com/off-the-shelf-datasets/>
- <https://www.speechdata.ai/guides/how-to-buy-speech-data> [ikkilamchi]

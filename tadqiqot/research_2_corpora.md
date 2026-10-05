# Tadqiqot 2: Haqiqiy nutq korpuslari qanday qurilgan + "noldan o'qitish" ning real narxi

Bu hujjat faqat ikki savolga javob beradi:
1. **A qism** — katta nutq korpuslari (LibriSpeech, LibriTTS, Emilia va h.k.) **aynan qanday** qurilgan: pipeline bosqichlari, segment qoidalari, filtr mezonlari, qancha audio **tashlab yuborilgan**.
2. **B qism** — TTS modelni **noldan (from scratch)** o'qitish uchun nashr etilgan real raqamlar: soat, GPU, kun; hamda fine-tuning bilan kesishish nuqtasi.

Alignment vositalarini taqqoslash, filtr chegaralarini chuqur tahlil qilish va o'zbek tiliga xos masalalar bu hujjatda **yo'q** (boshqa agentlar yopadi).

Har bir da'vo ostida URL bor. Manba topilmagan joyda ochiq **"ma'lumot topilmadi"** deb yozilgan.

---

## A QISM — Korpus qurish metodologiyasi

### A.1 LibriSpeech (Panayotov et al., ICASSP 2015) — foydalanuvchi g'oyasining eng yaqin pretsedenti

Manba: https://www.danielpovey.com/files/2015_icassp_librispeech.pdf · https://www.openslr.org/12/

Bu **aynan** foydalanuvchi qilmoqchi bo'lgan ish: "audiokitob + matni ma'lum → tekshirilgan, segmentlangan datasetga aylantirish". Pipeline quyidagicha bo'lgan:

**1-bosqich: audio ↔ matnni juftlash (kitob topish)**
- LibriVox API + Internet Archive metadata + Project Gutenberg RDF/XML fayllari solishtirilgan.
- Aniq sarlavha mos kelmaganda **sarlavhalarni "fuzzy matching"** qilingan (qamrovni oshirish uchun).
- Ko'p o'quvchili boblar chiqarib tashlangan: janr filtri + **LIUM toolkit** bilan avtomatik speaker diarization.

**2-bosqich: birinchi o'tish (qo'pol joylashtirish)**
- Bob audio avval **30 daqiqagacha** bo'laklarga kesilgan.
- Kaldi `gmm-decode-faster`, **VoxForge**da o'qitilgan triphone Boosted-MMI model bilan dekodlangan.
- Tanilgan matn va kitob matni orasida **Smith-Waterman** alignment ishlatilib, eng mos keladigan hudud topilgan.
- "**Islands of confidence**" (kamida 12 fonemali aniq moslik) bo'linish nuqtasi sifatida ishlatilgan.
- Natijada **35 sekunddan qisqa** segmentlar hosil qilingan.

**3-bosqich: ikkinchi o'tish (nozik moslashtirish + xatoni topish)**
- Transkript + **fonema darajasidagi bigram LM** dan maxsus decoding graph qurilgan — bu matn/audio nomuvofiqligini aniqlash uchun.
- Speaker-adapted model, **fMLLR** transformlar bilan.
- Decoder modifikatsiya qilingan: so'zlar orasiga insertion, ma'lum pozitsiyalarda substitution/deletion ruxsat etilgan — "faqat bitta elementni saqlashi mumkin bo'lgan pushdown transducer" kabi.

**Segment kesish qoidalari**
| Nima | Qoida |
|---|---|
| Bob boshlang'ich bo'lagi | ≤ 30 daqiqa |
| Birinchi alignmentdan keyin | ≤ 35 sekund |
| Train ma'lumot | **≥ 0.3 s sukunat** joyidan kesiladi |
| Test ma'lumot | faqat **gap (sentence) chegarasi**dan kesiladi |

**Qancha audio TASHLANDI va nega**
- Alignment natijasida **~1200 soat** hosil bo'lgan, yakuniy korpus **1000 soat** → **~200 soat (~17%) tashlab yuborilgan**.
- Sabablari (paperda sanab o'tilgan): Project Gutenberg matnining noaniqligi; o'quvchining insertion/deletion/substitution/transposition qilishi; disfluency (duduqlanish, takror); text normalization xatolari; grapheme-to-phoneme (G2P) xatolari.
- Muallliflar ochiq yozadi: "audiokitoblardagi audio hajmi shunchalik kattaki, ma'lum foizni yo'qotishga imkoniyatimiz bor" — ya'ni **strategiya: shubhali joyni tuzatmaslik, o'chirib tashlash**.

**WER asosidagi filtrlash va "clean" vs "other" bo'linishi**
- **WSJ si-84** akustik modeli + kitob matnidan qurilgan bigram LM bilan LibriSpeech audiosi tanilgan.
- Har bir speaker uchun avtomatik transkript va referens transkript orasidagi **WER** hisoblangan.
- Speakerlar WER bo'yicha saralanib, **taxminan o'rtadan** bo'lingan: past WER → `clean`, yuqori WER → `other`.
- `other` dev/test to'plamlari ataylab saralangan ro'yxatning **uchinchi kvartilidan** olingan (qiyinroq bo'lishi uchun).
- **Aniq WER chegarasi paperda berilmagan** — faqat nisbiy saralash.

**Yakuniy statistika**
| Subset | Soat | Speaker/daqiqa | Ayol | Erkak | Jami speaker |
|---|---|---|---|---|---|
| dev-clean | 5.4 | 8 | 20 | 20 | 40 |
| test-clean | 5.4 | 8 | 20 | 20 | 40 |
| dev-other | 5.3 | 10 | 16 | 17 | 33 |
| test-other | 5.1 | 10 | 17 | 16 | 33 |
| train-clean-100 | 100.6 | 25 | 125 | 126 | 251 |
| train-clean-360 | 363.6 | 25 | 439 | 482 | 921 |
| train-other-500 | 496.7 | 30 | 564 | 602 | 1166 |

- Sample rate: **16 kHz**. Gender balansi speaker soni va soat hajmi bo'yicha ta'minlangan.
- **Diqqat: har bir speakerga 25–30 daqiqa chegara qo'yilgan.** Bitta odamdan 100 soat olinmagan — ataylab ko'p speakerga tarqatilgan. Bu ASR uchun mantiqiy, TTS uchun esa teskari.

> **Foydalanuvchi uchun asosiy xulosa:** LibriSpeech pipeline'ining yuragi — *ikki o'tishli decode + Smith-Waterman + "ishonch orollari"* va **xato joyni tashlab yuborish**. Matn ma'lum bo'lsa ham, 17% audio ishga yaramadi.

---

### A.2 Libri-Light (Kahn et al., ICASSP 2020) — 60k soatga masshtablash

Manba: https://arxiv.org/abs/1912.07875 · https://ar5iv.labs.arxiv.org/html/1912.07875

- Jami **62.2K soat** (dublikatlar bilan).
- Subsetlar:

| Subset | Soat | Kitob | Speaker |
|---|---|---|---|
| unlab-60k | 57,706.4 | 9,860 | 7,439 |
| unlab-6k | 5,770.7 | 1,106 | 1,742 |
| unlab-600 | 577.2 | 202 | 489 |

- Format: **16 kHz FLAC**.
- Tozalash: buzilgan fayllar, speaker noma'lum yoki ko'p speakerli fayllar olib tashlangan; LibriSpeech dev/test speakerlari chiqarilgan.
- **Dublikat kitob versiyalari alohida ~4,500 soatlik subsetga ajratilgan** (ya'ni bir kitobning ikki o'qilishi train'ga tushmasligi uchun).
- **VAD:** 100M parametrli TDS akustik modeli, `wav2letter++` bilan frame-by-frame inference; harflar → SPEECH, sukunat → NONSPEECH; stride 8 frame (~10 ms segmentlar).
- **SNR baholash:** VAD posteriorlari orqali — posterior < 0.8 → speech, > 0.995 → noise (stabillik uchun speech frame'lariga keyingi 2 frame qo'shilgan).
- Metadata JSON: speaker ID, SNR, janr (7 kategoriya), VAD onset/offset, perplexity.
- Cheklangan supervision subsetlari (LibriSpeech'dan): **train-10h** (10 soat, 24 speaker, 12F/12M), **train-1h** (1 soat, 24 speaker), **train-10m** (6 ta 10-daqiqalik variant).

**Muhim nuqta:** Libri-Light'da **matn yo'q** — bu faqat audio + VAD segmentlari. Ya'ni "60k soat" o'z-o'zidan TTS dataset emas.

---

### A.3 Libriheavy (Kang et al., ICASSP 2024) — Libri-Light'ni matn bilan juftlash

Manba: https://arxiv.org/abs/2309.08105 · https://arxiv.org/html/2309.08105v2

Bu Libri-Light'ning 60k soatiga **matn biriktirish** ishi — foydalanuvchi uchun juda dolzarb, chunki bu "uzun audio + uzun matn" masalasining zamonaviy, ochiq-kodli yechimi.

**Pipeline:**
1. **Kitob matnini yuklash** — LibriVox metadatasidagi URL orqali. Oddiy tozalash: ortiqcha bo'shliq va satrlarni olib tashlash.
2. **Birinchi transkripsiya** — uzun audio **30 sekundli, har tomonda 2 sekund overlap** bilan bo'laklarga kesilib, LibriSpeech'da o'qitilgan ASR model bilan tanilgan. So'z timestamplari saqlangan va butun audio uchun birlashtirilgan.
3. **Matnda joy topish — suffix array** (klassik alignment emas!). Avtomatik transkript (query) va kitob matni (target) bitta uzun ketma-ketlikka ulanib, Kärkkäinen et al. (2006) algoritmi bilan suffix array qurilgan. "Close match"lar (i,j) koordinata juftlarini beradi.
4. **Longest increasing pairs** — monotonlikni ta'minlash uchun eng uzun o'suvchi juftliklar zanjiri topilgan: i₁≤i₂≤…≤iₙ va j₁≤j₂≤…≤jₙ.
5. **Levenshtein refinement** — nozik alignment, lekin **faqat (i,j) nuqtalari belgilagan bloklar ichida** (hisoblash hajmini kamaytirish uchun).

**Segment (cut) siyosati — TTS uchun juda muhim namuna:**
- Segment uzunligi **2–30 sekund**, **afzal ko'rilgani 5–20 sekund**.
- **Bo'linish faqat gap oxirini bildiruvchi punktuatsiyada**: `.`, `?`, `!`.
- Scoring funksiyasi birlashtiradi: sukunat davomiyligi (maksimum 3 sekund), insertion/deletion/substitution soni, davomiylik bonusi, moslik bonusi, xato jarimasi.
- Har bir BOS (Begin Of Segment) uchun **4 ta eng yaxshi EOS** nomzodi baholanadi.

**Punktuatsiya va katta/kichik harfni tiklash:** alohida model yo'q — **original kitob matnidan to'g'ridan-to'g'ri olinadi**. Segmentlar Lhotse "cut" formatida, `begin_byte`/`end_byte` atributlari bilan saqlanadi, shuning uchun kitobdan xohlagancha kontekst kesib olish mumkin. Bu LibriSpeech'ning normalizatsiya qilingan, punktuatsiyasiz matnidan asosiy farq.

**Statistika:**
| Subset | Soat | Kitob | Speaker | Speaker/soat |
|---|---|---|---|---|
| Small | 509 | 173 | 417 | 1.22 |
| Medium | 5,042 | 960 | 1,531 | 3.29 |
| Large | 50,794 | 8,592 | 6,736 | 7.54 |
| Dev | 22.3 | 180 | 141 | 0.16 |
| Test-clean | 10.5 | 87 | 70 | 0.15 |
| Test-other | 11.5 | 112 | 72 | 0.16 |

**Rad etish darajasi:** paper ochiq foiz bermaydi. Mening o'z arifmetikam: unlab-60k = 57,706 soat → Large = 50,794 soat, ya'ni **~88% saqlangan, ~12% yo'qotilgan**. Bu mening hisobim, paperning da'vosi emas.

---

### A.4 Multilingual LibriSpeech (MLS) (Pratap et al., Interspeech 2020)

Manba: https://arxiv.org/abs/2012.03411 · https://ar5iv.labs.arxiv.org/html/2012.03411

**Pipeline (4 bosqich):**
1. **Audio segmentatsiya** — uzun LibriVox fayllari ASG loss bilan o'qitilgan akustik modellar yordamida **10–20 sekundli** segmentlarga bo'linadi; kesish nuqtasi shu oraliq ichidagi sukunatdan tanlanadi.
2. **Pseudo-label** — **4-gram LM bilan beam-search decoding** orqali dastlabki transkript.
3. **Matn topish** — **TF-IDF** bilan nomzod kitoblarni topish + **Smith-Waterman** alignment; nomzod transkriptlar **WER > 40%** bo'lsa tashlanadi.
4. **Post-processing** — raqamlar pseudo-label alignmentiga qarab almashtiriladi; tire va apostroflar tilga xos heuristikalar bilan aytilgan shaklga moslashtiriladi.

**Statistika:**
| Til | Train (soat) | Dev | Test |
|---|---|---|---|
| English | 44,659.74 | 15.75 | 15.55 |
| German | 1,966.51 | 14.28 | 14.29 |
| Dutch | 1,554.24 | 12.76 | 12.76 |
| French | 1,076.58 | 10.07 | 10.07 |
| Spanish | 917.68 | 9.99 | 10.00 |
| Italian | 247.38 | 5.18 | 5.27 |
| Portuguese | 160.96 | 3.64 | 3.74 |
| Polish | 103.65 | 2.08 | 2.14 |

- Sample rate: original LibriVox 48 kHz → **16 kHz**ga tushirilgan.
- Speaker soni: Polyakcha 11 dan Inglizcha 5,490 gacha. Dev/test gender balansli, train/dev/test orasida speaker takrorlanmaydi.
- **Sifat filtridan keyin ~50.7K soatdan ~36.5K soat qolgan** (ya'ni **~28% rad etilgan**).
- Dev/test inson tomonidan tekshirilgan: avtomatik va inson transkripti orasidagi WER **1.93% (ispancha) – 12.64% (portugalcha)**. Ya'ni **hatto qo'lda tekshirilgan test to'plamida ham matn xatolari bor**.

---

### A.5 LJSpeech — bitta odamning o'qishi standart datasetga qanday aylandi

Manba: https://keithito.com/LJ-Speech-Dataset/

Bu foydalanuvchining loyihasi uchun **aniq maqsad (target)**: bitta odam, kitob o'qiydi, natija — TTS dataset.

| Ko'rsatkich | Qiymat |
|---|---|
| Klip soni | **13,100** |
| Jami davomiylik | **23 soat 55 daqiqa 17 sekund** (~24 soat) |
| Klip uzunligi | **1.11 – 10.10 sekund** |
| O'rtacha klip | **6.57 sekund** |
| Sample rate | **22,050 Hz** |
| Bit depth | 16-bit PCM, mono WAV |
| Hajm | 2.6 GB |
| Speaker | 1 (Linda Johnson) |
| Manba | LibriVox'dan **7 ta public domain non-fiction kitob** (1884–1964 nashrlari) |
| Yozib olish davri | 2016–17 |

**Metodologiya:**
- Klip **sukunatlar bo'yicha avtomatik segmentlangan** ("segmented automatically based on silences in the recording").
- Matnda ikki maydon: original va **normalized transcription** — raqamlar, tartib sonlar, pul birliklari to'liq so'zga yozilgan.
- **QA (qo'lda tekshirish) o'tkazilgan**: matn aytilganiga mos kelishi tasdiqlangan.
- Qisqartmalar (Mr., Mrs., Dr.) uchun alohida jadval berilgan.

**Hujjatlashtirilgan kamchiliklar:**
- Manba LibriVox MP3 bo'lgani uchun **MP3 kodlash artefaktlari** bo'lishi mumkin.
- **19 ta transkriptda ASCII bo'lmagan belgilar** bor.

> **Eng muhim raqam butun hujjatda: ~24 soat, 13,100 klip, 1–10 sekund, bitta speaker. Bu "noldan o'qitiladigan single-speaker TTS dataset" ning de-fakto standarti.**

---

### A.6 LibriTTS (Zen et al., Interspeech 2019) — ASR korpusidan TTS korpusi qanday yasaladi

Manba: https://arxiv.org/abs/1904.02882 · https://ar5iv.labs.arxiv.org/html/1904.02882

**Bu bo'lim foydalanuvchi uchun eng qimmatli:** TTS ASRdan qo'shimcha nima talab qiladi?

| Jihat | LibriSpeech (ASR) | LibriTTS (TTS) |
|---|---|---|
| Sample rate | 16 kHz | **24 kHz** |
| Matn | BOSH HARF, punktuatsiyasiz | **original VA normalizatsiyalangan matn ikkisi ham** |
| Segmentatsiya | >0.3 s sukunatdan kesish | **gap chegarasidan kesish** |
| Kontekst | tashlanadi | **qo'shni gaplar saqlanadi** |
| Shovqin filtri | faqat WER asosida saralash | **SNR filtri qo'shilgan** |

**TTS uchun qo'shilgan bosqichlar:**
1. Kitob matni → paragraf → gap (**Google sentence splitter**).
2. **Text normalization**: non-standard words va semiotic klasslar **weighted finite state transducer** bilan normallashtirilgan.
3. ASR transkripsiya: **Google Cloud Speech-to-Text**.
4. Force-alignment: modifikatsiyalangan **YouTube auto-sync** engine, LSTM akustik model.
5. **Qabul qilish mezoni juda qattiq:** gap "aligned" deb belgilanadi **agar barcha so'zlar mos kelsa (edit distance = 0)**.
6. **71 so'zdan uzun** gaplar chiqariladi.
7. **O'rtacha so'z davomiyligi katta** bo'lgan utterancelar chiqariladi (audio/matn nomuvofiqligi belgisi).
8. **WADA (waveform amplitude distribution analysis)** bilan SNR hisoblanadi.
   - **clean subset: SNR ≥ 20 dB**
   - **other subset: SNR ≥ 0 dB**
9. **Polarity normalization** (faza inversiyasini tuzatish) va **silence end-pointing** (boshi/oxiridagi sukunatni kesish).

**Filtr statistikasi (rad etish miqdori):**
| Subset | Boshlang'ich gap | Align bo'lmadi | Juda uzun | O'rtacha so'z davomiyligi | SNR filtri | Yakuniy |
|---|---|---|---|---|---|---|
| train-clean-360 | 262,107 | 78,058 | 1,805 | 26 | 65,718 | 116,500 |
| train-other-500 | 332,816 | 125,806 | 1,409 | 72 | 485 | 205,321 |

- `train-clean-360` da **gaplarning ~30%i alignment qilinmadi** va yana **~25%i SNR (20 dB) filtridan o'tolmadi**.
- Yakuniy korpus LibriSpeech audiosining **taxminan 60%i**.

**Yakuniy statistika:** **585 soat, 2,456 speaker, 24 kHz**.
| Subset | Soat | Ayol | Erkak | Jami |
|---|---|---|---|---|
| dev-clean | 8.97 | 20 | 20 | 40 |
| test-clean | 8.56 | 19 | 20 | 39 |
| dev-other | 6.43 | 16 | 17 | 33 |
| test-other | 6.69 | 17 | 16 | 33 |
| train-clean-100 | 53.78 | 123 | 124 | 247 |
| train-clean-360 | 191.29 | 430 | 474 | 904 |
| train-other-500 | 310.08 | 560 | 600 | 1,160 |

> **TTS ASRdan ortiq talab qiladigan narsalar (LibriTTS'dan chiqadigan ro'yxat):**
> 1. Yuqori sample rate (16 → 24 kHz va yuqoriroq)
> 2. Punktuatsiya + katta/kichik harf (original matn)
> 3. Gap chegarasidan kesish (sukunatdan emas)
> 4. **Nol xatolik alignment** (edit distance = 0), ASR'dagi "40–75% WERga toqat qilish" emas
> 5. SNR/shovqin filtri
> 6. Polarity va sukunat end-pointing
> 7. Anomal so'z davomiyligini aniqlash

---

### A.7 LibriTTS-R (Koizumi et al., Interspeech 2023) — "LibriTTS'ning o'zi ham yetarli toza emas edi"

Manba: https://arxiv.org/abs/2305.18802 · https://ar5iv.labs.arxiv.org/html/2305.18802

Bu **hujjatlashtirilgan muvaffaqiyatsizlik hikoyasi**: LibriTTS yuqoridagi barcha filtrlardan o'tgan bo'lsa ham, TTS uchun **yetarlicha toza bo'lmadi**.

- Muammo: LibriTTS sifati "LJSpeech kabi kichikroq lekin yuqori sifatli TTS datasetlar darajasida emas edi."
- Dalil: Glow-TTS LibriTTS'da faqat **3.45 MOS** bergan, ground-truth esa **4.22**.
- Yechim: **Miipher** speech restoration modeli (text-informed parametric re-synthesis): w2v-BERT feature → DF-Conformer feature-cleaner → **WaveFit** neural vocoder.
- Natija: **585 soat, 2,456 speaker, 24 kHz** (hajm o'zgarmagan, faqat audio tiklangan).
- Ground-truth MOS: test-clean **4.36 → 4.41**, test-other **3.94 → 4.09** (kichik o'sish).
- **TTS modelda o'sish katta:** LibriTTS-R train-960'da o'qitilgan model **4.06–4.31 MOS**, LibriTTS'da o'qitilgani esa **2.59–2.97 MOS**.

**Restoration'ning o'zi ham buzdi:** mualliflar tan oladi — **640 ta LibriTTS-R namunasidan 2 tasi SR muvaffaqiyatsizligi tufayli buzilgan**, va ular barcha training namunalarini tekshirib chiqmaganlarini aytadi.

> **Xulosa:** "shovqinli TTS dataset" = past MOS. 2.6 vs 4.1 MOS farqi faqat audio sifati tufayli. Bu foydalanuvchi uchun eng kuchli argument: **yozib olish sifati arxitekturadan muhimroq**.

---

### A.8 Hi-Fi TTS (Bakhturina et al., Interspeech 2021) — maksimal sifatga qaratilgan LibriVox pipeline

Manba: https://arxiv.org/abs/2104.01497 · https://ar5iv.labs.arxiv.org/html/2104.01497

Bu pipeline foydalanuvchi uchun eng yaqin **"TTS sifatida"** namuna, chunki maqsad ASR emas, TTS bo'lgan.

**Speaker tanlash:** ≥50 soat audiosi bo'lgan **250 ta LibriVox o'quvchisi**dan boshlangan → **10 speaker (6 ayol, 4 erkak)** qolgan.

**Audio sifat mezonlari:**
- **Bandwidth ≥ 13 kHz** (pikka nisbatan −50 dB da o'lchangan)
- **SNR ≥ 32 dB** (0.3–4 kHz diapazonda), `clean` subset uchun **≥ 40 dB**
- Sample rate: **44.1 kHz** (ZIP arxiv emas, alohida bob linklaridan yuklab olingan — ZIP versiyalari past bitrate bo'ladi)
- Tahlil usuli: birinchi 30 sekundda energiyaga asoslangan VAD

**Segmentatsiya va tekshirish:**
- **CTC-based segmentation**, **QuartzNet** ASR modeli bilan (LibriSpeech + Common Voice + WSJ + Fisher + Switchboard + Singapore English NSC'da o'qitilgan).
- Forward pass: harf log-ehtimolliklari; backward pass: optimal alignment yo'li.
- **Ishonch chegarasi: log fazoda −2 dan past segmentlar tashlanadi.**
- **Ikkinchi tekshirish:** QuartzNet VA Citrinet modellari greedy decoding bilan. **"Only samples with zero WER are included in the dataset"** — ya'ni **faqat WER = 0 bo'lgan segmentlar qoldirilgan**.
- Utterance uzunligi: **~20 sekund maksimum** (60-belgili chegara orqali), gap/punktuatsiya chegarasidan kesish.

**Statistika:** jami **291.6 soat** (115.4 soat clean + 176.2 soat other), **10 speaker**, har biriga **17.7–58.0 soat**.

Rad etish darajasi foizda berilmagan (250 o'quvchi → 10 speaker dan boshqa raqam **ma'lumot topilmadi**).

> **Diqqat: har bir speakerga 17.7–58 soat.** Ya'ni "yuqori sifatli bitta speaker TTS" uchun professional guruh ham **18+ soatdan** boshlagan.

---

### A.9 GigaSpeech (Chen et al., Interspeech 2021)

Manba: https://arxiv.org/abs/2106.06909 · https://ar5iv.labs.arxiv.org/html/2106.06909

- Manbalar: **audiokitob, podcast, YouTube**, 24 mavzu kategoriyasi. O'qilgan + spontan nutq.
- Aligner: **Kaldi asosida divide-and-conquer** — audio va transkript bo'laklanadi, biased LM bilan dekodlanadi, **TF-IDF** bilan moslanadi, **Smith-Waterman** bilan alignment qilinadi.
- **Segmentatsiya qoidalari:**
  - **>1 sekund sukunat**da kesish
  - **>0.2 sekund** punktuatsiya (`,` `.` `!` `?`) da kesish
  - **alignment WER ≥ 75%** bo'lgan segmentlar o'chiriladi
  - **≥ 20 sekund** segmentlar o'chiriladi
  - chegara sukunati **0.15 sekund**ga qisqartiriladi
- **Validation:** "forced alignment graph" — n-gram + leaky arcs, transkripsiya xatolarini aniqlash uchun.
- **Filtr qattiqligi:** XL subset (10,000 soat) uchun podcast/YouTube kontentida **WER chegarasi 4%**. Boshqa barcha subsetlar va audiokitoblar uchun **0% WER** (ya'ni hech qanday transkripsiya xatosi ruxsat etilmaydi).
- Subsetlar: **XS 10h, S 250h, M 1,000h, L 2,500h, XL 10,000h**.
- Format: 16 kHz, mono, 16-bit; Opus 32 kbps (WER ta'siri 0.1–0.2%).
- **Kirish vs chiqish:** "**~40,000 soat** transkriptlangan audio yig'ilgan" → XL = 10,000 soat. **Paper yakuniy rad etish foizini ochiq aytmaydi**; 40k → 10k arifmetikasi mening hisobim (~75% yo'qotish).

---

### A.10 The People's Speech (Galvez et al., NeurIPS 2021) — katta va hujjatlashtirilgan darajada shovqinli

Manba: https://arxiv.org/abs/2111.09344 · https://ar5iv.labs.arxiv.org/html/2111.09344

- **52,500 soat** manba → **30,000 soat** foydalanish mumkin bo'lgan audio.
- Manba: **faqat archive.org** (YouTube ToS sababli chiqarib tashlangan).
- Litsenziyalar: US Government works 28,034 soat; Public Domain 11,713 soat; CC-BY-SA 8,273 soat; CC-BY 4,496 soat.
- **Pipeline:**
  1. Kaldi GPU decoder + oldindan o'qitilgan akustik model bilan transkripsiya
  2. Hipoteza transkriptni **~15 sekundli** bo'laklarga bo'lish
  3. Har bir bo'lakda **DSAlign** forced alignment algoritmini ishlatish
- Hisoblash: **4 × NVIDIA T4**, 52,500 soatni ~52.5 soatda.
- **Filtr: CER > 50% bo'lsa segment rad etiladi** → **~21,100 soat olib tashlangan**.
- Sample rate taqsimoti: asosan **44.1 kHz (23,503 soat)** va 48 kHz (17,733 soat).

**Hujjatlashtirilgan MUVAFFAQIYATSIZLIK:** 100 ta tasodifiy namunani qo'lda tekshirish natijasi:
- **34%** — mukammal (0% WER)
- **22%** — yaxshi (1–8% WER)
- **44%** — past sifat (8–35% WER)

Mualliflar: **"44% (taxminan 14,000 soat) inson WER'idan yuqori"**, sababi — **alignment'dagi chegara (boundary) aniqlash xatolari**.

> **Bu foydalanuvchi uchun eng aniq ogohlantirish:** CER < 50% kabi bo'sh filtr bilan 30,000 soat "dataset" yasash mumkin, lekin uning 44%i yaroqsiz. TTS uchun bu o'limga teng — ASR shovqinli labelga toqat qiladi, TTS esa yo'q.

---

### A.11 YODAS (Li et al., ASRU 2023) — 500k soat, lekin sifat qatlamlangan

Manba: https://arxiv.org/abs/2406.00899 · https://ar5iv.labs.arxiv.org/html/2406.00899

- **500,000+ soat**, **140 til** (manual subset) / 20 til (automatic subset).
- Subsetlar: **manual 86,400 soat**, **automatic 335,845 soat**, **unlabeled 144,174 soat**.
- Pipeline: 3 xil worker —
  - **keyword-based client**: Wikipedia dump'idan kalit so'z olib, YouTube'da Creative Commons filtri bilan qidiradi
  - **channel-based client**: kanal ichida aloqador videolarni kengaytiradi
  - **download client**: audioni **1 kanal, 24 kHz** da yuklab oladi + subtitrlarni
- Subtitr: **manual subtitrga ustunlik**. Bir nechta subtitr mavjud va til ID'lari ziddiyatli bo'lsa, video **unlabeled** deb belgilanadi.
- Filtr: oldindan o'qitilgan akustik modelning **CTC loss score**i; **alignment chegarasi 2.0**.
- **Hujjatlashtirilgan muammo:** automatic subsetda "**misaligned utterancelarning sezilarli qismi**" bor, odatda musiqa yoki fon shovqini bilan.
- **CER: manual 14.9% vs automatic 32.3%.** Automatic'da deletion xatolari ayniqsa yuqori: **26.6% vs 7.5%**.

---

### A.12 Common Voice (Ardila et al., LREC 2020) — crowdsourced "gapni o'qi" modeli

Manba: https://arxiv.org/abs/1912.06670 · https://ar5iv.labs.arxiv.org/html/1912.06670

**Gaplar qanday tanlanadi:**
- 500,000+ Wikipedia maqolasi bo'lgan tillar uchun: **Wikipediadan jamoa tomonidan yozilgan tilga xos rule-set** bilan gap ajratib olinadi.
- Qo'shimcha gaplar **Sentence Collector** orqali: avtomatik validatsiya — **gap uzunligi, chet alifbo, raqamlar** tekshiriladi.
- Har bir taklif **3 ta reviewerdan 2 tasi** tomonidan tasdiqlanishi kerak → keyin haftalik eksportga tushadi.

**Validatsiya ovoz berish bilan:**
- Har bir audio klipni **maksimum 3 ta hissadorchi** eshitadi.
- **2 up-vote → valid**; **2 down-vote → invalid**.
- Faqat valid kliplar rasmiy train/dev/test to'plamiga tushadi.

**Format:** mono, **16-bit MP3, 48 kHz**.
**2019 noyabr holati:** 2,500 soat, 38 til, 50,000+ hissadorchi.
Rad etish darajasi/sifat statistikasi paperda **ma'lumot topilmadi**.

---

### A.13 Common Voice → TTS: hujjatlashtirilgan MUVAFFAQIYATSIZLIK va uni tuzatish

Manba: https://arxiv.org/abs/2210.06370 ("Can we use Common Voice to train a Multi-Speaker TTS system?")

Common Voice'ning **TTS uchun** aynan nima yomon ekani hujjatlashtirilgan:
1. **Shovqin** — "elektromagnit shovqin yoki akustik shovqin: mouse click, past chastotali shovqin, fon gapiruvchilari va fon musiqasi"
2. **Past bandwidth** — "audio fayllar low-pass filtrlangan, cutoff chastotasi fayldan faylga o'zgaradi"
3. **Noto'g'ri talaffuz** — notanish so'zlar, ba'zan butunlay boshqa tilda gapirish
4. **Yozib olishdagi tarqoqlik** — turli mikrofonlar, xona akustikasi, gapirish tezligi

**Tuzatish usuli:** **WV-MOS** (wav2vec2.0 + MLP head asosidagi MOS bahochi) bilan speaker-daraja filtri. Har bir speakerning utterance-level WV-MOS o'rtachasi olinib, turli chegaralar (2.0–4.0) bilan saralangan.

**Natija:**
- Boshlang'ich: **2,015 soat, 633 speaker**.
- **WV-MOS ≥ 4.0** chegarasi: faqat **86.05 soat, 88 speaker** qolgan (~4% saqlangan!).
- Kengaytirilgan `WV-MOS≥4.0-all`: 230.75 soat, 4,469 speaker.
- **MOS: 3.69 (filtrlangan) vs 2.35 (filtrsiz baseline)** → **+1.34 MOS**, faqat filtr hisobiga.
- Filtrlangan Common Voice LibriTTS'dan **0.35 MOS** yaxshiroq.

> **Bu foydalanuvchi uchun asosiy dars:** 2,015 soatdan 86 soat qoldi va bu **yaxshi** natija berdi. **Ko'p va shovqinli** dan **kam va toza** afzal. TTS'da agressiv filtr — bu yo'qotish emas, yutuq.

---

### A.14 Emilia / Emilia-Large (Emilia-Pipe) — zamonaviy "in-the-wild" pipeline

Manba: https://arxiv.org/abs/2407.05361 · https://arxiv.org/abs/2501.15907 · https://arxiv.org/html/2501.15907v2

**Emilia-Pipe — 6 bosqich, har biri nimani rad etadi:**

| # | Bosqich | Vosita | Nima qiladi / rad etadi |
|---|---|---|---|
| 1 | **Standardization** | — | WAV, mono, **24 kHz**, 16-bit, **−20 dBFS**ga normalizatsiya (gain ±3 dB bilan cheklangan) |
| 2 | **Source Separation** | **UVR-MDX-Net Inst 3** (Ultimate Vocal Remover) | fon musiqasi va shovqinni ajratib tashlaydi, faqat vokal qoldiradi |
| 3 | **Speaker Diarization** | **pyannote/speaker-diarization-3.1** | uzun nutqni bitta-speakerli utterancelarga bo'ladi; ko'p speakerli joylar yo'qoladi |
| 4 | **Fine-grained VAD segmentation** | **Silero-VAD** | ketma-ket voice-active bo'laklarni ulab, **3–30 sekundli** chunklar yasaydi |
| 5 | **ASR** | **Whisper-medium** (WhisperX orqali) | transkript hosil qiladi (matn allaqachon ma'lum bo'lgan holatda bu bosqich boshqacha bo'ladi) |
| 6 | **Filtering** | — | (a) **til aniqlash ishonchi < 80%** → rad; (b) **DNSMOS P.835 OVRL < 3.0** → rad; (c) **o'rtacha fonema davomiyligi 1.5× IQR** tashqarisida → rad |

**Rad etish darajasi — ikki xil raqam (MANBALAR FARQ QILADI):**

*v1 paper (arXiv 2407.05361):* 600-soatlik testda **598.87 soat → 176.22 soat**, ya'ni **29.43% saqlangan / ~70.6% rad etilgan**.

*Journal versiyasi (arXiv 2501.15907):* 666.94 soatlik testda bosqichma-bosqich:

| Bosqich | Chiqish (soat) | Saqlangan |
|---|---|---|
| Source Speech | 666.94 | 100% |
| + Source Separation | 666.94 | 100% |
| + Speaker Diarization | 515.50 | 77.29% |
| + VAD Segmentation | 522.03 | 78.27% |
| + ASR | 439.80 | 65.94% |
| + Filtering | **258.44** | **38.75%** |

Ikki raqam (29.43% va 38.75%) turli ishga tushirish/versiyaga tegishli — **ikkisini ham yozib qo'ydim, paperlar bir-biriga mos kelmaydi**.

**Real-Time Factor:** 0.006 — 666.94 soatni **8 × RTX 4090** da 240.5 daqiqada qayta ishlaydi.

**Hajm:**
| Til | Emilia (soat) | Emilia-Large (soat) | Kengayish |
|---|---|---|---|
| English | ~46.8k (46.77%) | **134.1k** (61.95%) | 2.9× |
| Chinese | ~49.9k (49.87%) | 57.3k (26.49%) | 1.1× |
| German | ~1.6k (1.59%) | 6.8k (3.15%) | 4.3× |
| French | ~1.4k (1.38%) | 8.2k (3.80%) | 6.0× |
| Japanese | ~1.7k (1.72%) | 2.5k (1.17%) | 1.5× |
| Korean | ~0.2k (0.22%) | 7.4k (3.44%) | 34.3× |
| **Jami** | **101,654** | **216,313** | |

- Emilia-Large qo'shimcha ma'lumot uchun **DNSMOS chegarasini 2.4**ga tushirgan (101k uchun 3.0 edi) — ya'ni masshtab uchun sifatdan voz kechilgan.
- Emilia-Large asosiy manbasi: **YODAS2**.

**Emilia (in-the-wild) vs MLS (audiokitob) taqqoslashi — muhim natija:**
- ~46k soat Emilia-En vs ~45k soat MLS'da AR+SoundStorm va VoiceBox o'qitilgan.
- **Rasmiy o'qish uslubida (LibriSpeech-Test): ikkisi ham deyarli teng.**
- **Spontan nutqda (Emilia-Test): Emilia yaqqol ustun** — AR+SoundStorm WER 7.7% (MLS) → 6.6% (Emilia); Speaker Similarity 0.587 → 0.618. VoiceBox SIM 0.528 → 0.601; FSD 15.94 → 14.07.

> **Foydalanuvchi uchun tarjima:** audiokitob ma'lumoti **o'qish uslubidagi** TTS uchun yetarli va MLS darajasida. Spontan/suhbat uslubi kerak bo'lsa, audiokitob yetmaydi.

---

### A.15 VCTK — studiyada yozilgan ko'p speakerli standart

Manba: https://datashare.ed.ac.uk/handle/10283/3443

- **110 ta inglizcha speaker**, turli aksentlar.
- Har bir speakerga **~400 gap**.
- Matn manbalari: **Herald Glasgow** gazetasi, **rainbow passage**, **speech accent archive**dan elicitation paragraph.
- Mikrofonlar: **DPA 4035** (omni) va **Sennheiser MKH 800** (small diaphragm condenser).
- Yozib olish: **96 kHz / 24-bit**, Edinburgh universitetining **hemi-anechoic** kamerasida.
- Chiqarilgan versiya: **48 kHz / 16-bit**, hammasi **qo'lda end-point qilingan**.
- Transkript 110 speakerdan **109 tasi** uchun mavjud (p315 matni disk xatosi tufayli yo'qolgan); p280 va p315'da MKH 800 yozuvlarida texnik muammo bor.
- **Jami soat datashare sahifasida ko'rsatilmagan.** VITS paperi VCTK'ni "~44,000 klip, 109 speaker, ~44 soat" deb beradi (https://ar5iv.labs.arxiv.org/html/2106.06103).

> **VCTK — bu "matn + audio align qilish" emas, boshqa model:** matn oldin yozilgan, odam studiyada o'qigan, qo'lda end-point qilingan. Foydalanuvchining g'oyasiga eng yaqin bo'lmagan, lekin eng yuqori sifatli yo'l.

---

### A.16 Whisper'ning ma'lumot pipeline'i (Radford et al., 2022) — 680k soat qanday filtrlangan

Manba: https://arxiv.org/abs/2212.04356 · https://ar5iv.labs.arxiv.org/html/2212.04356

Foydalanuvchining pipeline'i ASR ishlatadi, shuning uchun Whisper o'zi qanday ma'lumotda o'qitilganini bilish muhim (uning kuchli va zaif tomonlari shu yerdan keladi).

- **680,000 soat** internetdan yig'ilgan audio-transkript juftliklari; shundan **117,000 soat** 96 ta ingliz bo'lmagan tilda.
- **Mashina yozgan transkriptlarni olib tashlash heuristikalari:**
  - "Butunlay BOSH HARF yoki butunlay kichik harfdagi transkript inson yozgan bo'lishi juda ehtimoldan yiroq"
  - Murakkab punktuatsiya (`!`, `,`, `?`) yo'qligi
  - Katta/kichik harfda stilistik xilma-xillik yo'qligi
- **Til mosligini tekshirish:** **VoxLingua107**da fine-tune qilingan audio til detektori; transkript tili **CLD2** bilan aniqlanadi. Mos kelmasa — X→English juftliklari **tarjima ma'lumoti** sifatida qayta ishlatiladi, tashlanmaydi.
- **Fuzzy de-duplication** transkript matnlari bo'yicha.
- Xato darajasi va hajmi bo'yicha saralangan manbalarni **qo'lda ko'rib chiqish** — "qisman transkriptlangan yoki yomon align qilingan / mos kelmagan transkriptlar" topilgan va olib tashlangan.
- Train va evaluation to'plamlari orasida **transkript darajasida de-duplication** (TED-LIUM 3 uchun ayniqsa).

**Whisper'dan kelib chiqadigan xavf (muhim!):** ma'lumot qo'lda tekshirilmaganligi sababli, ma'noli nutq bo'lmagan audio bo'laklar tasodifiy matn bilan juftlangan, natijada model **non-speech → ravon matn** korrelyatsiyasini o'rgangan.
- Manba: https://arxiv.org/abs/2606.07473
- Miqdoriy dalil: **42 ta toza xona shovqini (room tone) klipida `whisper-large-v3` 61.9% hollarda so'z chiqaradi, 100% hollarda nimadir chiqaradi.** Manba: https://arxiv.org/abs/2609.32560
- Va shu paper muhim ogohlantirish beradi: hallucinationlarni korpusdan **filtrlab tashlash** "dalilni yo'q qiladi, lekin xatti-harakatni joyida qoldiradi" — student model o'qituvchi to'g'ri bo'lgan qismga moslanadi va o'z training ma'lumotida bo'lmagan nuqsonni meros qilib oladi.

> **Amaliy tavsiya:** ASR'ni **faqat alignment/tekshirish uchun** ishlatish kerak (matn allaqachon ma'lum!). ASR chiqishini **matn manbasi** sifatida ishlatish — bu Whisper hallucinationlarini datasetga kiritish demakdir.

---

### A.17 ParsVoice — audiokitobdan past resursli til uchun TTS korpusi (zamonaviy namuna)

Manba: https://arxiv.org/abs/2510.10774

Fors tili uchun audiokitoblardan avtomatik TTS korpusi. Foydalanuvchining g'oyasiga eng yaqin zamonaviy ish.

Pipeline komponentlari:
1. **Sentence completion** — fine-tune qilingan **ParsBERT** klassifikatori (gap tugallanganini aniqlash uchun)
2. **ASR asosida chegara (boundary) optimizatsiyasi**
3. **Punktuatsiya tiklash** (tilga xos)
4. **Avtomatik speaker ID** aniqlash
5. **Ko'p o'lchovli sifat bahosi** — audio va tilga xos matn xususiyatlari bo'yicha

Natija: **2,200 soat**, **1,815 speaker ID**, **1.36 million align qilingan juftlik**.
**XTTSv2** shu korpusda: **naturalness 3.6/5 MOS**, **speaker similarity 4.0/5 MOS**.
Sample rate, segment uzunligi siyosati va aniq rad etish foizi — abstraktda **ma'lumot topilmadi**.

---

### A.18 Umumiy korpus statistikasi jadvali

| Korpus | Soat | Speaker | Sample rate | Segment siyosati | Asosiy filtr | Rad etish |
|---|---|---|---|---|---|---|
| **LibriSpeech** | 1,000 | 2,484 (train) | 16 kHz | ≤35 s; train: ≥0.3 s sukunat; test: gap | 2-o'tishli decode + fonema bigram; WER bilan clean/other | ~200/1200 soat = **~17%** |
| **Libri-Light** | 62.2k (unlab-60k 57,706) | 7,439 | 16 kHz | TDS-VAD, ~10 ms stride | buzilgan/ko'p speakerli fayllar; dublikat kitoblar (~4.5k soat) ajratilgan | ma'lumot topilmadi |
| **Libriheavy** | 50,794 (large) | 6,736 | 16 kHz | **2–30 s (afzal 5–20 s)**, faqat `.?!` da kesish | suffix array + longest increasing pairs + Levenshtein; scoring | paper aytmaydi (~12%, mening hisobim) |
| **MLS** | 50.7k → **36.5k** | 5,490 (en) | 48 → 16 kHz | **10–20 s**, sukunatda | TF-IDF + Smith-Waterman; **WER > 40% rad** | **~28%** |
| **LJSpeech** | **23:55:17 (~24)** | **1** | **22.05 kHz** | **1.11–10.10 s (o'rtacha 6.57 s)**, sukunatda | qo'lda QA, matn tekshirilgan | ma'lumot topilmadi |
| **LibriTTS** | **585** | 2,456 | **24 kHz** | gap chegarasi; ≤71 so'z | **edit distance = 0**; WADA SNR ≥20 dB (clean) / ≥0 dB (other) | LibriSpeech audiosining ~40%i tashlangan |
| **LibriTTS-R** | 585 | 2,456 | 24 kHz | LibriTTS bilan bir xil | Miipher restoration | 640 dan 2 ta namuna buzilgan |
| **Hi-Fi TTS** | **291.6** (115.4 clean) | **10** (17.7–58 soat/speaker) | **44.1 kHz** | ≤~20 s, gap/punktuatsiya | bandwidth ≥13 kHz; SNR ≥32/40 dB; CTC score >−2; **WER = 0** | 250 o'quvchi → 10 speaker |
| **GigaSpeech** | 10,000 (XL) | ma'lumot topilmadi | 16 kHz | **<20 s**; >1 s sukunat; >0.2 s punktuatsiya | alignment WER ≥75% rad; XL: 4% WER; boshqa: **0% WER** | ~40k → 10k (mening hisobim ~75%) |
| **People's Speech** | 52.5k → **30,000** | ma'lumot topilmadi | 44.1/48 kHz | ~15 s chunk | **CER > 50% rad** | **21,100 soat (~40%)**; qolganning **44%i 8–35% WER** |
| **YODAS** | 500k+ (manual 86.4k) | ma'lumot topilmadi | 24 kHz | subtitr chegarasi | CTC alignment score chegara 2.0 | manual CER 14.9% / auto CER 32.3% |
| **Common Voice** | 2,500 (2019) | 50,000+ | 48 kHz MP3 | 1 gap = 1 klip | **2 up-vote / 2 down-vote** (max 3 tinglovchi) | ma'lumot topilmadi |
| **CV → TTS (curated)** | 2,015 → **86.05** | 633 → **88** | 48 kHz | — | **WV-MOS ≥ 4.0** | **~96% rad etilgan** |
| **Emilia** | **101,654** | ma'lumot topilmadi | **24 kHz** | **3–30 s** | DNSMOS ≥3.0; LID ≥80%; phone-duration 1.5×IQR | **61–71% rad** (v1: 29.43% saqlangan; journal: 38.75%) |
| **Emilia-Large** | **216,313** | ma'lumot topilmadi | 24 kHz | 3–30 s | DNSMOS **≥2.4** (pasaytirilgan) | ma'lumot topilmadi |
| **VCTK** | ~44 (VITS paperiga ko'ra) | **110** (~400 gap/speaker) | **96 kHz/24-bit → 48 kHz/16-bit** | 1 gap = 1 fayl, **qo'lda end-point** | studiya yozuvi, filtr kerak emas | — |
| **Whisper train** | 680,000 | — | — | 30 s oyna | mashina-transkript heuristikalari, LID moslik, fuzzy dedup | ma'lumot topilmadi |
| **ParsVoice** | 2,200 | 1,815 | ma'lumot topilmadi | ma'lumot topilmadi | ParsBERT sentence-completion + ko'p o'lchovli sifat | ma'lumot topilmadi |

---

## B QISM — "Noldan o'qitish" reallik tekshiruvi

### B.1 Modellar aynan nimada o'qitilgan (paperlardan real raqamlar)

| Model | Dataset | Soat | GPU | Batch | Steps/Epochs | O'qitish vaqti | Manba |
|---|---|---|---|---|---|---|---|
| **Tacotron 2** | ichki US English, 1 professional ayol | **24.6 soat** | **1 GPU** (feature net) / **32 GPU** (WaveNet vocoder) | 64 / 128 | ma'lumot topilmadi | ma'lumot topilmadi | https://ar5iv.labs.arxiv.org/html/1712.05884 |
| **FastSpeech 2** | LJSpeech | **~24 soat** (13,100 klip; 12,228 train) | **1 × V100** | 48 gap | **160,000 step** | **17.02 soat** | https://ar5iv.labs.arxiv.org/html/2006.04558 |
| FastSpeech 2s (waveform) | LJSpeech | ~24 soat | 2 × V100 | 6/GPU | 600,000 step | ma'lumot topilmadi | shu yerda |
| *Taqqoslash* | Transformer TTS | ~24 soat | — | — | — | **38.64 soat** | shu yerda |
| *Taqqoslash* | FastSpeech (teacher bilan) | ~24 soat | — | — | — | **53.12 soat** | shu yerda |
| **VITS** | LJSpeech | **~24 soat** (12,500 train / 100 val / 500 test) | **4 × V100** | 64/GPU (jami 256) | **800,000 step** | ma'lumot topilmadi | https://ar5iv.labs.arxiv.org/html/2106.06103 |
| **VITS (multi)** | VCTK | ~44 soat, 109 speaker | 4 × V100 | 64/GPU | 800,000 step | ma'lumot topilmadi | shu yerda |
| **Matcha-TTS** | LJSpeech | (paperda soat yozilmagan) | **2 GPU** (model **ma'lumot topilmadi**; eksperimentlarda RTX 3090) | 32 | **500,000 update** | **ma'lumot topilmadi** | https://ar5iv.labs.arxiv.org/html/2309.03199 |
| **StyleTTS 2** | LJSpeech | **24 soat** | **4 × A40** | 16 | 100 epoch pre-train + 60 epoch joint | ma'lumot topilmadi | https://ar5iv.labs.arxiv.org/html/2306.07691 |
| StyleTTS 2 | VCTK | **~44 soat**, 109 speaker | 4 × A40 | 16 | 50 + 40 epoch | ma'lumot topilmadi | shu yerda |
| StyleTTS 2 | LibriTTS train-clean-460 | **245 soat**, 1,151 speaker | 4 × A40 | 16 | 30 + 25 epoch | ma'lumot topilmadi | shu yerda |
| **F5-TTS (base)** | Emilia (en+zh) | **~95,000 soat** | **8 × A100 80G** | **307,200 audio frame ≈ 0.91 soat/batch** | **1.2M update** | **>1 hafta** | https://arxiv.org/html/2410.06885v2 |
| F5-TTS (small, ablation) | WenetSpeech4TTS Premium | **945 soat** (faqat mandarin) | 8 × RTX 3090 | base'ning yarmi | 800K update | 1 hafta | shu yerda |
| **XTTS** | 16 tilda ichki + ochiq | **27,281.6 soat** (en 14,513.1; pt 2,386.8; fr 2,215.5; ja 57.3; hu 62.0) | **4 × A100 80G** | 4 × grad.accum 16 | **~2.5M step** | ma'lumot topilmadi | https://ar5iv.labs.arxiv.org/html/2406.04904 |

**Muhim kuzatishlar:**
- **Barcha "klassik" single-speaker modellar (Tacotron 2, FastSpeech 2, VITS, StyleTTS 2, Matcha-TTS) LJSpeech'dagi ~24 soatda o'qitilgan.** 24 soat — tasodifiy son emas, bu de-fakto minimum-standart.
- StyleTTS 2 **245 soat** bilan zero-shot speaker adaptation qilgan va bu VALL-E'ning 60,000 soatidan **~250 marta kam** ekanini ta'kidlagan.
- **F5-TTS / XTTS kabi zamonaviy zero-shot modellarni noldan o'qitish 27,000–95,000 soat va 4–8 A100 talab qiladi.** Bu bitta odam uchun real emas.
- Matcha-TTS uchun sample rate va o'qitish vaqti paperda **ma'lumot topilmadi**.
- XTTS uchun "yangi til qo'shish uchun necha soat kerak" — paperda **ma'lumot topilmadi**.

### B.2 Nashr etilgan "necha soat kerak" da'volari

| Da'vo | Qiymat | Manba |
|---|---|---|
| "TTS modellar odatda bitta speaker uchun **10–20 soat** transkriptlangan nutqda o'qitiladi" | 10–20 soat | https://arxiv.org/abs/2110.05798 |
| Tacotron 2 amalda ishlatgani | **24.6 soat** | https://ar5iv.labs.arxiv.org/html/1712.05884 |
| LJSpeech (de-fakto standart) | **~24 soat** | https://keithito.com/LJ-Speech-Dataset/ |
| Hi-Fi TTS'da har bir speakerga | **17.7–58.0 soat** | https://ar5iv.labs.arxiv.org/html/2104.01497 |
| "**2 soatdan kam ma'lumot bilan, hatto noise augmentation bilan ham, ishlatishga yaroqli nutq chiqmaydi**" | **< 2 soat = ishlamaydi** | https://arxiv.org/abs/2306.10152 |
| Piper (VITS): noldan o'qitish uchun epoch | **~2000 epoch** noldan, fine-tuning uchun **+1000 epoch** | https://tderflinger.github.io/piper-docs/guides/training/ |

**MANBALAR FARQ QILADI:** 10–20 soat (2110.05798) vs 24.6 soat (Tacotron 2) vs 17.7–58 soat/speaker (Hi-Fi TTS). Konservativ o'qish: **minimum ishlaydigan chegara ~10 soat, xavfsiz maqsad 20–25 soat, yuqori sifat uchun 30–50 soat.**

### B.3 Noldan o'qitganda kam ma'lumot bilan AYNAN nima buziladi

Bu qism eng muhim, chunki muvaffaqiyatsizlik rejimi bitta va u har joyda takrorlanadi: **matn ↔ nutq alignment (attention) o'rganilmaydi.**

**1) Attention/alignment kollapsi**
- **LRSpeech** (https://ar5iv.labs.arxiv.org/html/2008.03687): faqat **50 namuna (5 daqiqa)** yuqori sifatli + **1,000 namuna (3.5 soat)** past sifatli paired ma'lumot bilan noldan o'qitilgan baseline'lar **"reasonable speech" sintez qila olmadi**. Intelligibility va MOS "ishlatishga yaroqsiz" deb belgilangan. **Word Coverage Ratio ~0.65**, attention weightlar tarqoq.
- Faqat boy resursli tillarda pre-training qilgandan keyin **"attention alignmentlar diagonal bo'lib qoldi"** — bu pre-training TTS'da matn↔nutq alignmentini qurishga yordam berishini ko'rsatadi.
- LRSpeech'ning to'liq tizimi yakunda: **intelligibility 98.08%**, **MOS 3.57** (ground truth'dan 0.48 farq).

**2) Tushunarsiz (garbled) nutq va >100% WER**
- https://arxiv.org/abs/2306.10152: 2 soatlik ma'lumotda, ularning usuli qo'llanmasa (tasodifiy tanlov, augmentation yo'q) — chiqish **tushunarsiz**, **WER > 133%**.
- Muallliflar: "tasodifiy low-resource o'qitish fonema va mel-spectrogram frame'lari orasidagi alignmentni o'rgana olmaydi, natijada ko'rilmagan matn uchun buzilgan chiqish beradi."
- Ularning usuli bilan: **2 soat → MOS 3.98±0.09**, 23.5 soatlik LJSpeech baseline **3.97±0.09** — statistik jihatdan teng. Lekin bu **augmentation + maxsus ma'lumot tanlash** bilan, oddiy noldan o'qitish emas.

**3) Zamonaviy flow-matching modellarda ham xuddi shu**
- **Parameter-Efficient Fine-Tuning for Low-Resource TTS** (Interspeech 2025, https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf):
  > "training the F5-TTS model from scratch in low-resource environments using only a single GPU and a small dataset is challenging, **often resulting in misalignment between text and speech**"
- Dataset: **KSS (Korean Single Speaker), 12.65 soat, 12,853 namuna**, bitta professional ayol ovozi. GPU: **1 × TITAN RTX**.

| Rejim | CER | WER | SIM-O | NMOS |
|---|---|---|---|---|
| **F5-TTS noldan (pre-training yo'q)** | **17.345** | **45.883** | 0.641 | **2.64 ± 0.41** |
| **F5-TTS fine-tuned (pre-trained)** | **5.671** | **12.032** | 0.659 | **3.58 ± 0.32** |
| Ularning PEFT usuli (5.81M trainable = 1.72%) | 4.906 | 12.625 | — | **3.88 ± 0.16** |

**Fine-tuning CER'ni 67%, WER'ni 74% kamaytirdi** — va bu **12.65 soatlik, professional, bitta speakerli, toza datasetda**. Ya'ni LJSpeech'ga taxminan teng hajmda ham, noldan o'qitish fine-tuningdan **keskin yomon** ishlagan.

**4) Amaliy tasdiq (community)**
- https://github.com/SWivid/F5-TTS/discussions/769: noldan o'qitish "shovqin sifatida boshlandi va faqat sekin nutqqa aylanib bordi", pre-trained modeldan fine-tuning esa ishlagan. 10–60 daqiqalik ma'lumot bilan **150 epoch (~30k step)** da sezilarli yaxshilanish. RTX 4070 laptop'da ~20k–25k step/kun.

**Muvaffaqiyatsizlik rejimlari xulosasi:**
| Ma'lumot hajmi | Nima bo'ladi |
|---|---|
| < 2 soat | Alignment umuman o'rganilmaydi; WER > 133%; tushunarsiz chiqish |
| 5 daqiqa + 3.5 soat (LRSpeech baseline) | "Reasonable speech" yo'q; Word Coverage Ratio ~0.65; tarqoq attention |
| ~12.65 soat (KSS, F5-TTS noldan) | Ishlaydi lekin yomon: CER 17.3%, WER 45.9%, NMOS 2.64 |
| ~24 soat (LJSpeech) | Barcha klassik modellar shu yerda ishlaydi (Tacotron2/FastSpeech2/VITS/StyleTTS2) |

### B.4 Kesishish nuqtasi: fine-tuning qachon noldan o'qitishdan yaxshi?

**Dalil 1 — TTS, speaker adaptatsiyasi (2110.05798):** 1 / 5 / 30 / 60 daqiqalik ma'lumot sinalgan.
| Rejim | MOS (Speaker 92) | MOS (Speaker 6097) |
|---|---|---|
| Real ovoz | 3.94 | 3.91 |
| **Noldan (27+ soat)** | **3.44** | **3.29** |
| **30 daqiqada fine-tuning** | **3.40** | **3.41** |

Xulosa (paperdan): "**faqat 30 daqiqalik** transkriptlangan audio bilan bizning usullarimiz **27+ soatda noldan o'qitilgan model** bilan naturalness, ovoz o'xshashligi va uslub o'xshashligi bo'yicha raqobatbardosh natija beradi." Bu **~54 barobar** kamroq ma'lumot.
Manba: https://arxiv.org/abs/2110.05798

**Dalil 2 — TTS, past resursli til (Kwon et al. 2025):** **12.65 soatda** fine-tuning noldan o'qitishdan **keskin** ustun (CER 17.3 → 5.7). Ya'ni TTS'da kesishish nuqtasi **12.65 soatdan ancha yuqori**.
Manba: https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf

**Dalil 3 — ASR (TTS emas, lekin aniq kesishish raqami bor):** Fong et al., https://arxiv.org/abs/2508.05149 — SLAM-ASR framework, Italyan tili, 10/15/20/50/100/200/252 soat sinalgan.
| Ma'lumot | Noldan (WER, CV IT) | Ingliz pre-trained | Ispan pre-trained |
|---|---|---|---|
| 10 soat | **14.0%** | **9.8%** | **8.6%** |
| 200 soat | **6.4%** | 6.6% | 6.7% |

Paperdan: "**fine-tuning ma'lumoti 100–200 soatga oshgani sari, pre-trained modelni fine-tune qilishning afzalligi yo'qoladi**." 200 soatda noldan o'qitilgan model pre-trained variantlar bilan **tenglashdi**.
Galisiy tili misolida 10 soatda: noldan 18.6% WER vs ko'p tilli pre-trained 13.3% WER.

> **Kesishish nuqtasi (mening sintezim, manbalar asosida):**
> - **< 2 soat** → noldan o'qitish **ishlamaydi**; faqat fine-tuning/adaptatsiya.
> - **2–20 soat** → fine-tuning **aniq yutadi** (F5-TTS 12.65 soatda: CER 17.3 vs 5.7).
> - **20–50 soat** → noldan o'qitish **texnik jihatdan mumkin** (LJSpeech = 24 soat), lekin fine-tuning hali ham tezroq va barqarorroq.
> - **~100–200+ soat** → noldan o'qitish pre-trained model bilan **tenglashadi** (ASR dalili; TTS uchun ekvivalent raqam **ma'lumot topilmadi**).

**MANBALAR ZIDDIYATI — muhim istisno:** hamma manba fine-tuning tarafida emas.
- **"Exploring Pre-training Benefits on Phoneme Addition through Fine-tuning in Speech Synthesis"** (https://arxiv.org/html/2606.19792v1): **yangi fonemalarni o'rganish** nuqtai nazaridan, **fine-tuning noldan o'qitishga teng yoki KO'PROQ ma'lumot talab qiladi**.
  - Sinalgan hajmlar: 100, 300, 500, 800, 1,000, 2,000 utterance. Model: Conformer-FastSpeech2 + HiFi-GAN. Metrika: wav2vec2.0 asosidagi PER, UTMOS.
  - Simulyatsiya (plosive undoshlar): noldan o'qitish barcha hajmlarda teng yoki yaxshiroq PER bergan.
  - Real cross-lingual (ingliz → yapon): "**training from scratch consistently achieved better Target PER than fine-tuning across all data sizes**".
  - Muallliflar: "ko'rilgan fonemalarni generatsiya qilish bo'yicha pre-trained qobiliyat ko'rilmagan target fonemalarni o'zlashtirish uchun cheklangan foyda beradi" — pre-training **naturalness**ga yordam bergan, lekin **fonema o'zlashtirishga** emas.

> **Bu ziddiyatning amaliy ma'nosi:** manba model tilida yo'q bo'lgan fonemalar/tovushlar ko'p bo'lsa, fine-tuning'ning afzalligi kamayadi. Lekin bu paper ham noldan o'qitish **kam ma'lumot** bilan ishlaydi demaydi — u faqat *fonema o'zlashtirish* o'qini o'lchaydi va naturalness'da pre-training foydali ekanini tasdiqlaydi.

### B.5 Hisoblash narxi — real raqamlar bilan taqqoslash

| Maqsad | Ma'lumot | GPU | Vaqt |
|---|---|---|---|
| FastSpeech 2 noldan (LJSpeech) | 24 soat | 1 × V100 | **17.02 soat** |
| VITS noldan (LJSpeech) | 24 soat | 4 × V100 | 800k step (vaqt: ma'lumot topilmadi) |
| StyleTTS 2 noldan (LJSpeech) | 24 soat | 4 × A40 | 160 epoch (vaqt: ma'lumot topilmadi) |
| F5-TTS noldan (Emilia) | 95,000 soat | 8 × A100 80G | **>1 hafta** |
| XTTS noldan | 27,281.6 soat | 4 × A100 80G | ~2.5M step (vaqt: ma'lumot topilmadi) |
| F5-TTS PEFT fine-tune (KSS) | 12.65 soat | **1 × TITAN RTX** | 355,000 step |

> **Oson o'qish:** "single-speaker TTS'ni noldan o'qitish" (FastSpeech2/VITS uslubi) **bitta GPU'da bir kunda** bajariladigan ish — **agar sizda 20+ soat toza ma'lumot bo'lsa**. "Zero-shot ko'p tilli foundation modelni noldan o'qitish" (F5-TTS/XTTS uslubi) — bu **8 × A100 × 1 hafta + 27k–95k soat** ish, ya'ni butunlay boshqa kategoriya.

---

## C — MUVAFFAQIYATSIZLIKLAR RO'YXATI (hujjatlashtirilgan)

Foydalanuvchi uchun eng qimmatli qism: kimlar yomon dataset yasagan va nega.

| # | Loyiha | Nima bo'ldi | Hujjatlashtirilgan sabab | Manba |
|---|---|---|---|---|
| 1 | **The People's Speech** | 30,000 soatlik datasetning **44%i (≈14,000 soat) insonning WER'idan yomon** (8–35% WER) | Filtr juda bo'sh (**CER > 50%** chegarasi); **alignment chegara (boundary) aniqlash xatolari** | https://arxiv.org/abs/2111.09344 |
| 2 | **LibriTTS** | TTS modellar unda **2.59–2.97 MOS** bergan (LibriTTS-R'da 4.06–4.31) | Audio shovqinli — WER/SNR filtridan o'tgan bo'lsa ham TTS uchun yetarli toza emas | https://arxiv.org/abs/2305.18802 |
| 3 | **LibriTTS-R (restoration)** | **640 namunadan 2 tasi restoration tufayli buzilgan**; mualliflar barcha training namunalarini tekshirmagan | Speech restoration modelining o'zi nuqson kiritadi | https://arxiv.org/abs/2305.18802 |
| 4 | **YODAS (automatic subset)** | **CER 32.3%** (manual 14.9%); "**misaligned utterancelarning sezilarli qismi**" | Avtomatik YouTube subtitrlari; musiqa/fon shovqini; deletion xatolari 26.6% | https://arxiv.org/abs/2406.00899 |
| 5 | **Common Voice (TTS uchun)** | Filtrsiz o'qitilgan TTS **2.35 MOS**; filtrlangan **3.69 MOS** | Crowdsourced shovqin, o'zgaruvchan low-pass filtr, noto'g'ri talaffuz, xilma-xil mikrofon; **validatsiya mezoni juda bo'sh** ("turli fon shovqinlari ruxsat etiladi") | https://arxiv.org/abs/2210.06370 |
| 6 | **Whisper training ma'lumoti** | Model **non-speech → ravon matn** korrelyatsiyasini o'rgangan; `whisper-large-v3` toza xona shovqinida **61.9% hollarda so'z chiqaradi** | Ma'lumot qo'lda tekshirilmagan; ma'nosiz audio tasodifiy matn bilan juftlangan | https://arxiv.org/abs/2606.07473 · https://arxiv.org/abs/2609.32560 |
| 7 | **Whisper pseudo-label'dan distillation** | Hallucinationlarni korpusdan filtrlab tashlash "**dalilni o'chiradi, xatti-harakatni joyida qoldiradi**" — student o'z training ma'lumotida bo'lmagan nuqsonni meros qiladi | Filtrlash tanlanma (selection bias) yaratadi | https://arxiv.org/abs/2609.32560 |
| 8 | **MLS dev/test (qo'lda tekshirilgan!)** | Inson va avtomatik transkript orasidagi WER **1.93%–12.64%** | Hatto qo'lda tekshirilgan test to'plamida ham matn xatolari qoladi | https://arxiv.org/abs/2012.03411 |
| 9 | **LJSpeech** | **19 ta transkriptda** ASCII bo'lmagan belgilar; MP3 kodlash artefaktlari | Manba LibriVox MP3; text pipeline to'liq tozalamagan | https://keithito.com/LJ-Speech-Dataset/ |
| 10 | **F5-TTS noldan, kam ma'lumotda** | **CER 17.3%, WER 45.9%, NMOS 2.64** (12.65 soatda) | "misalignment between text and speech" — bitta GPU + kichik dataset | https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf |
| 11 | **Tacotron 2 uslubi, 2 soatda, augmentationsiz** | **WER > 133%**, tushunarsiz chiqish | "fonema ↔ mel frame alignmenti o'rganilmaydi" | https://arxiv.org/abs/2306.10152 |
| 12 | **LRSpeech baseline'lari** | "Reasonable speech" umuman chiqmagan; Word Coverage Ratio ~0.65 | 5 daqiqa + 3.5 soat paired ma'lumot — attention diagonal bo'lmaydi | https://arxiv.org/abs/2008.03687 |

---

## D — Foydalanuvchining pipeline'i uchun bevosita xulosalar

**1. G'oya to'g'ri va pretsedenti bor.** "To'liq matn + uzun audio → TTS dataset" — bu aynan LibriSpeech (1000 soat), Libriheavy (50,794 soat), LibriTTS (585 soat), Hi-Fi TTS (291.6 soat) va ParsVoice (2,200 soat) qilgan ish. G'ildirakni qaytadan o'ylab topish kerak emas.

**2. Matn ma'lum bo'lishi katta yutuq, lekin sehrli tayoqcha emas.** LibriSpeech matn ma'lum bo'lgan holda ham **17% audioni** tashlab yubordi; LibriTTS `train-clean-360`da gaplarning **~30%i align bo'lmadi**. Sizning pipeline'ingiz **rad etish darajasi 15–40%** bo'lishini normal deb qabul qilishi kerak.

**3. TTS uchun ASR'dan farqli qattiqroq talablar (LibriTTS + Hi-Fi TTS'dan):**
   - 24 kHz yoki yuqori (Hi-Fi TTS: 44.1 kHz, bandwidth ≥13 kHz)
   - Punktuatsiya + katta/kichik harf original matndan (Libriheavy uslubi)
   - **Gap chegarasidan** kesish, sukunatdan emas
   - Qabul mezoni: **edit distance = 0** (LibriTTS) yoki **WER = 0** (Hi-Fi TTS) — ASR'ning 40–75% toleransi emas
   - SNR/DNSMOS filtri (LibriTTS: WADA SNR ≥20 dB; Emilia: DNSMOS ≥3.0)
   - Polarity normalization + silence end-pointing
   - Anomal so'z/fonema davomiyligini aniqlash

**4. Segment uzunligi — eng ko'p qo'llanilgan qiymatlar:**
   - LJSpeech: **1.11–10.10 s** (o'rtacha 6.57 s) — single-speaker TTS uchun eng yaqin namuna
   - Libriheavy: **2–30 s**, afzal **5–20 s**, faqat `.?!` da kesish
   - Emilia: **3–30 s**
   - MLS: **10–20 s**
   - GigaSpeech: **<20 s**
   - Hi-Fi TTS: **≤~20 s**

**5. ASR'ni matn manbasi sifatida ishlatmang.** Matn allaqachon bor — ASR faqat **joy topish va tekshirish** uchun. Aks holda Whisper hallucinationlari datasetga kiradi (toza xona shovqinida 61.9% hollarda so'z chiqaradi).

**6. "Ko'p" dan "toza" afzal.** Common Voice tajribasi: 2,015 soat → 86 soat (96% rad) → MOS 2.35 → 3.69.

---

*Hujjat tuzilgan sana: 2026-09-29. Barcha raqamlar yuqorida ko'rsatilgan URL'lardagi paperlardan olingan. Hisoblab chiqarilgan (paperda ochiq yozilmagan) raqamlar "mening hisobim" deb belgilangan.*

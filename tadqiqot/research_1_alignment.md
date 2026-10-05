# Tadqiqot 1 — Forced Alignment: uzun audioni MA'LUM matnga moslash

**Tadqiqot doirasi:** faqat forced alignment texnologiyasi. Sifat filtri, mavjud datasetlar va o'zbek tili xususiyatlari boshqa agentlar zimmasida.
**Sana:** 2026-09-29. Har bir da'vo ostida URL keltirilgan.
**Masala:** bitta to'liq matn (kitob/skript) + o'sha matnni o'qigan odamning uzun audiosi (soatlar) → avtomatik ravishda qisqa moslangan bo'laklar + transkript (LJSpeech formati).

---

## 0. Qisqa xulosa (TL;DR)

Zamonaviy amaliyot **CTC forced alignment** atrofida yig'ilgan: ASR akustik modelining CTC log-posterior chiqishini oling, ma'lum matnni token ketma-ketligiga aylantiring va Viterbi/trellis orqali eng ehtimolli yo'lni topib har bir belgi/so'zga vaqt bering. Kaldi/HMM (MFA, Gentle) va DTW (aeneas) avlodlari eskirdi — birinchisi yangi til uchun akustik model o'qitishni talab qiladi, ikkinchisi esa audio-matn mos kelmasligiga juda sezgir.

O'zbek tili uchun eng muhim fakt: **MMS/wav2vec2 asosidagi ko'p tilli char-level CTC aligner** (`torchaudio.pipelines.MMS_FA`, 1130 tilda 31K soatda o'qitilgan) uchun sizga o'zbek akustik modeli **kerak emas** — matnni `uroman` bilan lotinlashtirasiz va char-level CTC ishlaydi.

---

## 1. Umumiy solishtirish jadvali

| Vosita | Transkript kerakmi | Chiqish granularligi | Uzun audio strategiyasi | Til yo'q bo'lsa | Windows | Litsenziya | GPU |
|---|---|---|---|---|---|---|---|
| **aeneas** | Ha (majburiy) | Fragment (jumla/paragraf); word-level `mplain` bilan | Butun fayl RAM'ga; 4GB RAM ≈ 2h, 16GB ≈ 10h | 38 til ro'yxati bilan cheklangan (UZB yo'q) | Ishlaydi, lekin C-extension `cew` og'rig'i bor; all-in-one installer mavjud | AGPL-3.0 | Yo'q (CPU, C extension) |
| **MFA** | Ha (+ talaffuz lug'ati) | So'z **va** fonema (TextGrid) | Korpus allaqachon qisqa utterance'larga bo'lingan bo'lishi kerak | Rasmiy o'zbek modeli yo'q → o'zingiz `mfa train` qilasiz | `conda -c conda-forge` (Kaldi conda orqali); rasmiy hujjatda Windows aniq yozilmagan, Docker yo'li ko'rsatilgan | MIT (modellar CC BY 4.0) | Asosan CPU (ko'p jarayon) |
| **ctc-segmentation** (kutubxona) | Ha | Belgi (char) → utterance; so'z-darajasi har so'zni utterance qilib | `get_partitions()` bilan bo'lish | Har qanday CTC model bilan; char-level | Pip + Cython, muammosiz | Apache-2.0 | Model GPU'da |
| **torchaudio.functional.forced_align** + **MMS_FA** | Ha | Token (char) → so'z (`merge_tokens`, `TokenSpan`) | O'zi bo'lmaydi; qo'lda chunklash kerak (`batch_size==1`) | **1130+ til**, `uroman` lotinlashtirish | Toza pip, eng oson | torchaudio BSD-2; **MMS_FA vazn CC-BY-NC 4.0** | Ha, CUDA |
| **ctc-forced-aligner** (MahmoudAshraf97) | Ha | sentence / word / char (`--split_size`) | **Ichida qurilgan**: `--window_size 30` s + `--context_size 2` s | MMS-300m (1130+ til) + `--romanize` | Pip + FFMPEG, eng oson | BSD-2 (default model CC-BY-NC) | Ha, `--device cuda` |
| **fairseq MMS data_prep** | Ha (har satr = bitta segment) | Segment (satr) | `align_and_segment.py` butun faylni oladi | 1100+ til, uroman | `sox`/`apt install` bilan Linux'ga yo'naltirilgan | MIT (fairseq) | Ha |
| **NeMo Forced Aligner (NFA)** | Ixtiyoriy (`align_using_pred_text`) | token / word / **segment** CTM + ASS | "1+ soatli audio"da ishlatilishi mumkin | 14+ til NeMo checkpointlari; o'zbek CTC modeli kerak | NeMo Linux/Docker'ga yo'naltirilgan, Python 3.12+ | Apache-2.0 | Ha |
| **NeMo CTC-Segmentation tool** | Ha | Segment manifest + kesilgan audio | To'liq pipeline: prepare → segment → verify → cut | NeMo CTC modeli kerak | Linux/Docker | Apache-2.0 | Ha |
| **WhisperX** | Yarim (Python `align()` API) | So'z (+ char) | Whisper 30s oynalar + VAD | Torch: en/fr/de/es/it; HF: ~35 til; **o'zbek yo'q → ValueError** | `pip install whisperx` | BSD-2 | Ha, <8GB VRAM |
| **stable-ts** | Ha (`align()`, `--align`) | So'z | Whisper ichidagi oynalash | Whisper tillari | Pip | MIT | Ha |
| **whisper-timestamped** | **Yo'q** | So'z (cross-attention DTW) | Whisper segmentlari | Whisper tillari | Pip | AGPL-3.0 | Ha |
| **Gentle** | Ha | So'z (+ `case` holati) | Kaldi | **Faqat ingliz** | Native yo'q, Docker | MIT | Yo'q |
| **DSAlign** | Ha | Fragment (jumla) | VAD + rekursiv divide&conquer | DeepSpeech modeli kerak | ma'lumot topilmadi | MPL-2.0 | DeepSpeech |
| **audiogrep** | **Yo'q** | So'z/jumla (pocketsphinx) | — | Ingliz | Arxivlangan (2024-03) | MIT | Yo'q |
| **ReadAlongs Studio** | Ha | So'z (TextGrid/EAF/SRT/VTT) | ma'lumot topilmadi | **Akustik model shart emas** (g2p + SoundSwallower) | `pip install readalongs`, Windows uchun conda | MIT | Yo'q |

---

## 2. Har bir vosita bo'yicha batafsil

### 2.1 aeneas

**Manba:** https://github.com/readbeyond/aeneas , https://www.readbeyond.it/aeneas/docs/

1. **Kirish/chiqish.** Transkript **majburiy**. Kirish: audio + matn fayli (fragmentlar satrlarga bo'lingan). Chiqish "sync map" — juda ko'p format: `AUD, CSV, EAF, JSON, SMIL, SRT, SSV, SUB, TEXTGRID, TSV, TTML, TXT, VTT, XML`. Granularlik — fragment darajasi (bitta so'zdan paragrafgacha); so'z darajasidagi aniqlik uchun "multilevel text" (`is_text_type=mplain`), MFCC masking yoki yaxshiroq TTS engine kerak.

   Buyruq (README'dan aynan):
   ```
   python -m aeneas.tools.execute_task \
       audio.mp3 \
       text.txt \
       "task_language=eng|os_task_file_format=json|is_text_type=plain" \
       map.json
   ```

2. **Maksimal uzunlik.** Hujjatda aniq yozilgan: *"No protection against memory swapping: be sure your amount of RAM is adequate for the maximum duration of a single audio file (e.g., 4 GB RAM => max 2h audio; 16 GB RAM => max 10h audio)."* Ya'ni butun MFCC matritsasi RAM'da — bu asosiy chegara. Strategiya: audioni oldin bo'laklab, har boliga mos matn bo'lagini berish.

3. **Til.** 38 til: `AFR, ARA, BUL, CAT, CYM, CES, DAN, DEU, ELL, ENG, EPO, EST, FAS, FIN, FRA, GLE, GRC, HRV, HUN, ISL, ITA, JPN, LAT, LAV, LIT, NLD, NOR, RON, RUS, POL, POR, SLK, SPA, SRP, SWA, SWE, TUR, UKR`. **O'zbek (UZB) yo'q.** aeneas eSpeak (default), AWS Polly, Festival kabi TTS engine'lardan foydalanadi — ya'ni "akustik model" o'rniga TTS bilan sun'iy audio yasab, uni haqiqiy audio bilan DTW qiladi. eSpeak-ng'da `uz` ovozi bor-yo'qligini aeneas ro'yxatiga qo'shish mumkinligi haqida **ma'lumot topilmadi**.

4. **Aniqlik.** Kürzinger va h.k. (2020) TEDlium v2 ustida aeneas'ni (DTW) o'lchagan: **o'rtacha chetlanish 9.01 sekund** — ya'ni tekshirilgan sharoitda juda yomon. Manba: https://ar5iv.labs.arxiv.org/html/2007.09127 . ManaTTS maqolasi ham aeneas'ni *"audio-matn mos kelmasligiga sezgirligi sababli"* rad etgan (https://arxiv.org/html/2409.07259v1).

5. **Windows.** Rasmiy holat: "Debian 64-bit (official); Linux, macOS, Windows (confirmed working)" + all-in-one installer bor. Amalda og'riq real:
   - numpy aeneas'dan **oldin** o'rnatilishi kerak (setup.py C extension mexanizmi sababli);
   - `cew` C extension (TTS tezlashtirish) **faqat Linux**da;
   - Windows'da espeak bilan linking xatolari (unresolved external symbols) ma'lum muammo;
   - yechim: `AENEAS_WITH_CEW=False` yoki sillsdev/aeneas-installer.
   Manbalar: https://github.com/readbeyond/aeneas/issues/306 , https://github.com/ReadBeyond/aeneas/issues/315 , https://github.com/readbeyond/aeneas/blob/master/wiki/INSTALL.md , https://github.com/sillsdev/aeneas-installer/releases

6. **Litsenziya.** AGPL-3.0 — diqqat: yopiq/SaaS mahsulotga qo'shish uchun muammoli.

7. **GPU/tezlik.** GPU ishlatmaydi. MFCC+DTW Python C extension orqali. Aniq RTF raqami **ma'lumot topilmadi**.

---

### 2.2 Montreal Forced Aligner (MFA)

**Manba:** https://montreal-forced-aligner.readthedocs.io/ , https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner

1. **Kirish/chiqish.** Transkript **majburiy** + **talaffuz lug'ati** (pronunciation dictionary) majburiy. Kirish: korpus katalogi (`.wav` + bir xil nomli `.lab`/`.TextGrid`). Chiqish: Praat **TextGrid** — **so'z darajasi VA fonema darajasi** (bu MFA'ning asosiy afzalligi; boshqa hech bir ro'yxatdagi vosita fonema chegarasini bunday bermaydi).

2. **Maksimal uzunlik — MUHIM CHEGARA.** MFA soatlik uzluksiz audioni bo'lish uchun mo'ljallanmagan; u **allaqachon qisqa utterance'larga bo'lingan** korpusni kutadi. Ya'ni bizning masalada MFA **birinchi qadam bo'lolmaydi** — u faqat ikkinchi qadam (fonema darajasida aniqlashtirish) uchun yaroqli. Uzun audio uchun MFA `mfa align` emas, `mfa segment` yoki Kaldi'ning `steps/cleanup/segment_long_utterances.sh` kabi vositalar kerak bo'ladi.

3. **Til / yangi til qo'shish.** Oldindan o'qitilgan akustik modellar ro'yxati (https://mfa-models.readthedocs.io/en/latest/acoustic/index.html): Bulgarian, Croatian, Czech, English, French, German, Hausa, Japanese, Korean, Mandarin, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Thai, Turkish, Ukrainian, Vietnamese. **Rasmiy MFA o'zbek modeli yo'q** — faqat "Uzbek CV acoustic model v2_0_0" bor, u ham MFA phoneset emas, **XPF phoneset**dan foydalanadi. Modellar CC BY 4.0, CV modellari CC-0.

   Yangi til/akustik model o'qitish (https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/train_acoustic_model.html):
   ```
   mfa train [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH OUTPUT_MODEL_PATH
   ```
   Muhim opsiyalar: `--output_directory` (alignment eksporti), `--config_path`, `--phone_groups_path`, `--g2p_model_path` (G2P modelini qo'shish), `--language` (yapon/koreys/xitoy/tay tokenizatsiyasi). Hujjat: *"You don't need a ton of data to generate decent alignments"* — lekin **aniq minimal soat ma'lumot topilmadi**. Monophone→triphone→LDA→SAT bosqichlari bu sahifada aniq sanab o'tilmagan (**ma'lumot topilmadi**).

   **Tovuq-tuxum muammosi:** MFA modelini o'qitish uchun qisqa moslangan utterance'lar kerak — lekin biz ularni yasamoqchimiz. Demak MFA bizning pipeline'da faqat MMS/CTC bilan birinchi bo'lish qilingandan **keyin** ishlaydi.

4. **Aniqlik.** MFA 3.0 maqolasi: McAuliffe, Gunter, Wagner, Sonderegger (2026), *"Montreal Forced Aligner and the state of speech-to-text alignment in 2026"*, Interspeech 2026. Aniq ms xatolik raqamlari men o'qigan sahifalarda **ma'lumot topilmadi**.

5. **Windows.** **Manbalar ziddiyatli:**
   - MFA 2.1.7 hujjati: *"Prior to this version, G2P and language model training was unavailable on native Windows, but could be used with Windows Subsystem for Linux (WSL)"* — ya'ni 2.0.6'dan keyin hamma narsa hamma platformada. https://montreal-forced-aligner.readthedocs.io/en/v2.1.7/installation.html
   - MFA 3.x hujjatida (`docs/source/installation.rst`, main branch) Windows **faqat bir joyda** eslatiladi — Docker yo'l formati haqida: *"For Windows, the path style is `//c/Users/path`"*. Source install bo'limi Kaldi/OpenFst/OpenGRM-ngram/Baumwelch/Pynini'ni "for Mac/Linux" deb belgilaydi. https://raw.githubusercontent.com/MontrealCorpusTools/Montreal-Forced-Aligner/main/docs/source/installation.rst
   - Tarixiy Windows conda muammosi: #270 `PackagesNotFoundError` — `baumwelch, ngram, pynini, openfst` win-64 kanalida yo'q. https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/270
   - `kalpy` (MFA 3.x'ning Kaldi bindingi) uchun win-64 build mavjudligini tasdiqlay olmadim — **ma'lumot topilmadi**. Amaliy tavsiya: Windows'da MFA'ni **Docker** yoki **WSL2** orqali ishlatishga tayyor bo'ling.

   O'rnatish (aynan):
   ```
   conda create -n aligner -c conda-forge montreal-forced-aligner -y
   conda activate aligner
   ```
   ```
   docker image pull mmcauliffe/montreal-forced-aligner:latest
   docker run -it -v /path/to/data/directory:/data mmcauliffe/montreal-forced-aligner:latest
   ```
   conda-forge oxirgi versiya: **3.4.2**, yangilangan **2026-08-20** (https://anaconda.org/conda-forge/montreal-forced-aligner).

6. **Litsenziya.** MIT (kod), modellar CC BY 4.0 — tijorat uchun qulay.

7. **GPU/tezlik.** Kaldi asosan CPU, ko'p jarayonli. GPU faqat SpeechBrain (VAD/diarizatsiya) qismlari uchun. Aniq RTF **ma'lumot topilmadi**.

---

### 2.3 ctc-segmentation (Kürzinger va h.k., 2020) — "klassik" yechim

**Manbalar:** https://github.com/lumaku/ctc-segmentation , https://arxiv.org/abs/2007.09127 (v1 2020-07-17, v2 2020-10-05), https://ar5iv.labs.arxiv.org/html/2007.09127

1. **Kirish/chiqish.** Kirish: (a) audio, (b) CTC **log-posterior** matritsasi (o'zingiz modeldan olasiz), (c) ma'lum transkript, (d) model lug'atidan belgi ro'yxati. Chiqish: har utterance uchun `start`, `end` va **confidence score**, hamda belgi darajasidagi ehtimollik moslashuvi.

   API funksiyalari:
   - `prepare_token_list()` / `prepare_text()` — ground truth → token matritsasi
   - `ctc_segmentation()` — CTC log-posteriorlardan **char-wise** alignment
   - `determine_utterance_segments()` — char alignment → utterance segmentlari
   - `CtcSegmentationParameters` — konfiguratsiya klassi
   - `get_partitions()` — uzun audioni bo'lish

   So'z darajasidagi vaqtni olish uchun **har so'zni alohida "utterance" qilib** berish kerak (char-level model bilan).

2. **Algoritm (maqoladan aynan).** Trellis:
   ```
   k[t,j] = max( k[t-1,j] * p(blank|t) ,  k[t-1,j-1] * p(c_j|t) )    t>0, j>0
   k[t,j] = 0   agar t=0 va j>0
   k[t,j] = 1   agar j=0
   ```
   Backtracking oxirgi belgining eng ehtimolli vaqt pozitsiyasidan boshlanadi.

   **Murakkablikni kamaytirish:** O(M·N) → **O(M)**, "audio va matn pozitsiyasi nisbati deyarli doimiy" evristikasi asosida faqat `[t − W/2, t + W/2]` oynasidagi kadrlar hisoblanadi. **Bu aynan uzun audio muammosining yechimi** — butun trellis emas, faqat diagonal lenta.

3. **Confidence score.** Moslangan kadrlar L uzunlikdagi bo'laklarga bo'linadi, har bo'lakda o'rtacha ehtimollik hisoblanadi, keyin **eng kichigi** olinadi (`score_min_mean_over_L` parametri). Maqola: bu formula *"inflicts a penalty on the confidence score on mismatch, e.g., even if a single word is missing"* — ya'ni bitta so'z tushib qolsa ham ball keskin tushadi. **Bu bizning "o'qigan odam so'z tashlab ketdi" muammosini aniqlash uchun tayyor mexanizm.**

4. **Aniqlik raqamlari (TEDlium v2, maqoladan).**

   | Aligner | O'rtacha chetlanish | Std | ±0.5 s ichida |
   |---|---|---|---|
   | **CTC-segmentation** (Transformer) | **0.31 s** | 0.85 s | **88.8%** |
   | Gentle (HMM) | 0.41 s | ma'lumot topilmadi | ma'lumot topilmadi |
   | aeneas (DTW) | 9.01 s | ma'lumot topilmadi | ma'lumot topilmadi |

   **Tavsiya etilgan threshold:** confidence `s_seg < 0.22` (log fazoda **−1.5**) bo'lgan alignmentlar tashlab yuborilgan.

   Bu maqola nemis tili uchun 1700+ soat korpusda ishlatilib, Tuda-DE test setida **12.8% WER** (oldingi hybrid DNN/HMM baseline 14.4%).

5. **Windows.** `pip install ctc-segmentation`. Manbadan qurish Cython kompilyatsiyasini talab qiladi. Kaldi kerak emas, Docker kerak emas. **Windows uchun eng oson toifadagi vositalardan.**

6. **Litsenziya.** Apache-2.0 (tijorat uchun toza).

7. **GPU/tezlik.** Algoritmning o'zi CPU/NumPy-Cython; og'ir qism — CTC posteriorlarni hisoblovchi model (GPU'da). README'da wav2vec2 bilan to'liq misol keltirilgan. Aniq RTF **ma'lumot topilmadi**.

---

### 2.4 torchaudio.functional.forced_align + MMS_FA (rasmiy PyTorch yo'li)

**Manbalar:**
- API: https://docs.pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html
- API tutorial: https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
- Ko'p tilli tutorial: https://docs.pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html
- MMS_FA bundle: https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.MMS_FA.html

1. **Kirish/chiqish.** Aniq signatura:
   ```python
   torchaudio.functional.forced_align(
       log_probs: Tensor,          # (B, T, C) CTC emission log-probability
       targets: Tensor,            # (B, L) tokenlashtirilgan transkript
       input_lengths: Optional[Tensor] = None,   # (B,)
       target_lengths: Optional[Tensor] = None,  # (B,)
       blank: int = 0
   ) -> Tuple[Tensor, Tensor]      # (har kadr uchun label, har kadr uchun log-score)
   ```
   Keyin `merge_tokens()` takrorlanishlarni birlashtirib **`TokenSpan`** obyektlari beradi: maydonlari `token`, `start` (kadr indeksi), `end` (exclusive), `score` (span bo'yicha o'rtacha ehtimollik). So'z darajasiga o'tish uchun token spanlarni so'z chegaralari bo'yicha guruhlaysiz.

   Yuqori darajali pipeline:
   ```python
   from torchaudio.pipelines import MMS_FA as bundle
   model = bundle.get_model()
   tokenizer = bundle.get_tokenizer()
   aligner = bundle.get_aligner()

   def compute_alignments(waveform, transcript):
       with torch.inference_mode():
           emission, _ = model(waveform.to(device))
           token_spans = aligner(emission[0], tokenizer(transcript))
       return emission, token_spans
   ```

2. **Maksimal uzunlik — ASOSIY CHEKLOV.** Hujjatda aniq yozilgan: *"The current version only supports batch_size==1"*. Shuningdek `log_probs` uzunligi ≥ target uzunligi + ketma-ket takrorlangan tokenlar soni bo'lishi kerak. Tutorialdagi misollar 2.5–4.6 s bo'laklar. **Soatlik audio uchun o'zingiz chunklash yozishingiz kerak** — bu MMS `data_prep` va `ctc-forced-aligner` paketlari hal qilgan narsa (pastda).

   > Rasmiy torchaudio tutoriallarida uzun (soatlik) audio uchun aniq strategiya berilmagan — **ma'lumot topilmadi**.

3. **Til — bu yerda hal qiluvchi afzallik.** `MMS_FA`: *"Trained on 31K hours of data in 1,130 languages"* (Pratap va h.k., 2023, "Scaling Speech Technology to 1,000+ Languages"). Boshqa Wav2Vec2 bundle'lardan farqi: **so'z chegarasi tokeni (`|`) yo'q** — post-processing shunga mos bo'lishi kerak.

   Matn tayyorlash (o'zbek uchun ham xuddi shu):
   - **Romanizatsiya** (lotin bo'lmagan yozuv uchun) — `uroman`;
   - belgi normalizatsiyasi (kichik harf, punktuatsiyani olib tashlash);
   - xitoy kabi tillar uchun so'zga bo'lish.
   ```python
   def normalize_uroman(text):
       text = text.lower()
       text = text.replace("’", "'")
       text = re.sub("([^a-z' ])", " ", text)
       text = re.sub(' +', ' ', text)
       return text.strip()
   ```
   O'zbek lotin alifbosi allaqachon `a-z` + `'` ga juda yaqin (`o'`, `g'`, `sh`, `ch`, `ng`) — ya'ni **uroman'siz ham deyarli to'g'ridan-to'g'ri ishlaydi**.

   **`<star>` tokeni:** MMS_FA modelida default holda qo'shimcha `<star>` emission o'lchovi bor; `get_model(with_star=False)` bilan o'chirish mumkin. Vazifasi: *"Handles partial or mismatched transcripts by modeling unspecified content as wildcard tokens, improving alignment accuracy when transcript deletions exist."* — **ya'ni matnda yo'q, lekin audioda bor narsani (yo'talish, "uh", takroriy jumla) `<star>` yutib yuboradi.** Bu bizning masalada juda qimmatli.

4. **Aniqlik.** torchaudio hujjatlarida MMS_FA uchun aniq ms xatolik raqamlari **ma'lumot topilmadi**. Tutorialda CTC'ning ma'lum kamchiligi eslatilgan: *"peaky behavior"* — CTC label paydo bo'lganda spike beradi, lekin labelning to'liq davomiyligini qoplamaydi, shu sababli segment chegaralari noaniq bo'ladi; blank tokenning talqini kontekstga qarab o'zgaradi va so'z chegarasi aniqligiga ta'sir qiladi.

5. **Windows.** Eng oson: `pip install torch torchaudio` (CUDA wheel). Kaldi yo'q, Docker yo'q, WSL yo'q, tashqi binarlar yo'q (faqat ffmpeg/soundfile audio o'qish uchun). **Windows 11 + GPU uchun eng kam og'riqli variant.**

6. **Litsenziya — DIQQAT.** torchaudio kodi BSD-2, lekin **MMS_FA model vaznlari `CC-BY-NC 4.0`** — ya'ni **tijorat ishlatishga ruxsat bermaydi**. Agar yasalgan dataset tijoriy mahsulotda ishlatilsa, bu huquqiy savol tug'diradi (alignment vositasining litsenziyasi natijaviy datasetga o'tadimi — bu munozarali, lekin xavfni hisobga olish kerak). Alternativ: o'z wav2vec2 CTC modelini o'qitish yoki Apache/MIT litsenziyali XLSR fine-tune ishlatish.

7. **GPU/tezlik.** CUDA'da ishlaydi. `batch_size==1` cheklovi tufayli throughput chunklash bilan hal qilinadi. Rasmiy RTF raqami **ma'lumot topilmadi**.

---

### 2.5 ctc-forced-aligner (MahmoudAshraf97) — amaliy o'ram

**Manbalar:** https://github.com/MahmoudAshraf97/ctc-forced-aligner , https://pypi.org/project/ctc-forced-aligner/ , https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner

Bu — torchaudio CTC forced alignment + MMS modelining **uzun audio uchun tayyor, chunklashi ichida qurilgan** o'rami. Bizning masala uchun eng to'g'ri "tayyor" vosita.

1. **Kirish/chiqish.** Transkript majburiy. Granularlikni siz tanlaysiz: `--split_size` ∈ {`sentence`, `word`, `char`} (default `word`). Chiqish: JSON (`text` + `segments` ro'yxati `start`/`end`/`text` bilan). README'ga ko'ra HuggingFace Wav2Vec2/HuBERT/MMS modellarini qo'llaydi va **"at least 5X less memory usage" than TorchAudio's implementation**.

2. **Uzun audio — ichida qurilgan.** `--window_size 30` (sekund) oynalar, `--context_size 2` (sekund) overlap. Ya'ni soatlik audio 30 s bo'laklarga, 2 s kesishuv bilan bo'linadi va emissionlar ketma-ket hisoblanadi. `--batch_size 4` (default).

   To'liq argument ro'yxati (README'dan):

   | Argument | Ma'no | Default |
   |---|---|---|
   | `--audio_path` | audio fayl | majburiy |
   | `--text_path` | matn fayl | majburiy |
   | `--language` | ISO 639-3 kodi (o'zbek: `uzb`) | majburiy |
   | `--romanize` | lotin bo'lmagan yozuv uchun | False |
   | `--split_size` | `sentence`/`word`/`char` | `word` |
   | `--star_frequency` | `segment` yoki `edges` | `edges` |
   | `--merge_threshold` | segmentlarni birlashtirish | 0.00 |
   | `--alignment_model` | model nomi | `MahmoudAshraf/mms-300m-1130-forced-aligner` |
   | `--compute_dtype` | inference dtype | `float32` |
   | `--batch_size` | batch | 4 |
   | `--window_size` | chunk (s) | 30 |
   | `--context_size` | overlap (s) | 2 |
   | `--attn_implementation` | attention turi | `eager` |
   | `--device` | `cuda`/`cpu` | avtomatik |

   Buyruq:
   ```bash
   ctc-forced-aligner --audio_path "path/to/audio.wav" --text_path "path/to/text.txt" --language "eng" --romanize
   ```

3. **Til.** Default model MMS-300m — **1130+ til**, `--romanize` bilan. O'zbek uchun akustik model o'qitish kerak emas.

4. **Aniqlik.** README'da aniq raqam **ma'lumot topilmadi** (asos MMS_FA + torchaudio CTC forced align, demak 2.3-bo'limdagi CTC-segmentation sinfidagi aniqlik kutiladi, lekin bu ekstrapolyatsiya, o'lchangan raqam emas).

5. **Windows.** `pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git` + FFMPEG. Kaldi/Docker/WSL kerak emas. **Windows'da muammosiz toifada.**

6. **Litsenziya.** Kod **BSD-2**; **default model CC-BY-NC 4.0** — README'da aniq ogohlantirish bor: tijorat uchun boshqa model ishlatish kerak.

7. **GPU/tezlik.** GPU'ni avtomatik aniqlaydi. Aniq RTF **ma'lumot topilmadi**.

---

### 2.6 fairseq MMS `data_prep` (Facebook/Meta rasmiy dataset yasash vositasi)

**Manba:** https://github.com/facebookresearch/fairseq/tree/main/examples/mms/data_prep

Bu — **aynan bizning masala** uchun Meta tomonidan chiqarilgan referens pipeline: uzun audio + transkript → qisqa segmentlar + manifest.

Qadamlar (README'dan aynan):
```bash
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
git clone git@github.com:isi-nlp/uroman.git
apt install sox
pip install sox dataclasses
```
Matn fayli: **har satr = bitta kerakli audio segment**. Keyin:
```bash
python align_and_segment.py --audio /path/to/audio.wav --text_filepath /path/to/textfile \
    --lang <iso> --outdir /path/to/output --uroman /path/to/uroman/bin
```

**Chiqish:** FLAC segmentlar + `manifest.json`, maydonlari:
- `audio_filepath` — segment yo'li
- `duration` — davomiylik (s)
- `text` — asl transkript
- `normalized_text` — tozalangan transkript
- `uroman_tokens` — bo'sh joy bilan ajratilgan belgi tokenlari
- `audio_start_sec` — asl audiodagi boshlanish vaqti

**Texnik detallar** (`align_utils.py` dan): `stride_msec = 20` → `frames_per_sec = 1000 / stride_msec = 50` kadr/sekund (ya'ni vaqt rezolyutsiyasi **20 ms**). wav2vec2 `aux_num_out=31` token chiqishi.

**Segment chegaralarini sukunatga kengaytirish (muhim nozik detal):** `get_spans(tokens, segments)` funksiyasi ikki segment orasidagi jimlikni **ikkiga bo'lib** taqsimlaydi:
```python
pad_start = prev_seg.start if (idx == 0) else int((prev_seg.start + prev_seg.end)/2)
```
Ya'ni har segment o'zidan oldingi pauzaning yarmini oladi — bu **so'z o'rtasidan kesib yuborishni** va boshlanishdagi tovushni yo'qotishni oldini oladi. Bu juda oddiy, lekin amalda ishlaydigan qoida.

**Windows:** README `apt install sox` deb Linux'ga yo'naltirilgan; nightly torchaudio talabi ham noqulay. Amalda `ctc-forced-aligner` (2.5) — shu g'oyaning Windows'ga qulay qayta yozilishi.

**Litsenziya:** fairseq MIT; MMS model vaznlari CC-BY-NC 4.0.

---

### 2.7 NeMo Forced Aligner (NFA) — NVIDIA

**Manba:** https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html

1. **Kirish/chiqish.** Kirish: NeMo JSON manifest — `audio_filepath` (absolyut yo'l) + ixtiyoriy `text`. Model: CTC yoki Hybrid CTC-Transducer (**pure Transducer qo'llanmaydi**).
   ```bash
   python <path_to_NeMo>/tools/nemo_forced_aligner/align.py \
       pretrained_name="stt_en_fastconformer_hybrid_large_pc" \
       manifest_filepath=<path_to_manifest> \
       output_dir=<output_path>
   ```
   **Chiqish — uch darajali CTM** (bu bizning masalada juda qulay):
   - `<output_dir>/ctm/tokens/<utt_id>.ctm`
   - `<output_dir>/ctm/words/<utt_id>.ctm`
   - `<output_dir>/ctm/segments/<utt_id>.ctm`

   Format: `<utt_id> 1 <start_seconds> <duration_seconds> <text>`
   Qo'shimcha: ASS subtitr fayllari (token va word darajasida), `ASSFileConfig` bilan sozlanadi.

2. **Uzun audio.** Hujjatda aniq: *"NFA can be used on long audio files of 1+ hours duration (subject to your hardware and the ASR model used)"*. Manifest'da `duration` maydoni talab qilinmaydi. Aniq maksimal uzunlik ko'rsatilmagan — xotira bilan cheklangan.

3. **Segmentlash — juda muhim parametr.** `additional_segment_grouping_separator` — matnni kichik segmentlarga ajratuvchi satrlar ro'yxati, **default `['.', '?', '!', '...']`**. Ya'ni NFA "segment" darajasi = jumla darajasi, va CTM segment fayli to'g'ridan-to'g'ri jumla-darajasidagi kesish nuqtalarini beradi. Bu bizning "jumla chegarasida kes" talabini bevosita qondiradi.

   Boshqa parametrlar: `align_using_pred_text` (True bo'lsa audio ASR bilan transkripsiya qilinadi va **shu** matnga moslanadi — default False), `batch_size` (default 1), `transcribe_device` / `viterbi_device`, `remove_blank_tokens_from_ctm` (default False), `minimum_timestamp_duration`, `use_buffered_chunked_streaming`.

   Hujjat ogohlantiradi: *"NFA will also produce better alignments the more accurate the reference text in `text` is"*.

4. **Til.** "14+ languages" mavjud NeMo ASR checkpointlari orqali, lekin *"only English tested extensively"*. **O'zbek uchun NeMo CTC checkpointi kerak** — ya'ni o'zingiz o'qitishingiz kerak. Bu NFA'ni bizning boshlang'ich holat uchun mos qilmaydi.

5. **Aniqlik.** Aniq ms xatolik raqami **ma'lumot topilmadi**.

6. **Windows.** NeMo hujjati Linux va Docker'ga yo'naltirilgan; **Python 3.12+** talab qiladi; Windows yoki WSL rasmiy hujjatda eslatilmagan (**ma'lumot topilmadi** — ya'ni rasmiy Windows qo'llovi yo'q deb hisoblash xavfsiz). Amalda WSL2 + Docker tavsiya etiladi.

7. **Litsenziya.** Apache-2.0 (https://github.com/NVIDIA/NeMo). GPU: *"NVIDIA GPU + CUDA (required for training; recommended for inference)"*.

---

### 2.8 NeMo "CTC-Segmentation" dataset yasash vositasi — REFERENS PIPELINE

**Manba:** https://github.com/NVIDIA/NeMo/tree/main/tools/ctc_segmentation , https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/ctc_segmentation.html

Hujjat ta'rifi: *"This tool provides functionality to align long audio files and the corresponding transcripts into shorter fragments that are suitable for an Automatic Speech Recognition (ASR) model training."* — bizning masalaning aynan o'zi (ASR uchun, lekin TTS uchun ham xuddi shu shakl).

**To'rt bosqichli pipeline** (`run_segmentation.sh` dan):
1. `python prepare_data.py` — matn/audio normalizatsiyasi (tilga xos)
2. `python run_ctc_segmentation.py` — **ikki xil oyna uzunligi bilan ikki marta** ishga tushiriladi
3. `python verify_segments.py` — moslangan segmentlarni tekshiradi
4. `python cut_audio.py` — confidence threshold bo'yicha filtrlaydi va yakuniy manifest yasaydi

**`run_segmentation.sh` default qiymatlari (aynan):**
```
MIN_SCORE=-2
OFFSET=0
LANGUAGE='en'
MAX_SEGMENT_LEN=30
SAMPLE_RATE=16000
MAX_DURATION=20
ADDITIONAL_SPLIT_SYMBOLS=":|;"
USE_NEMO_NORMALIZATION='True'
NUM_JOBS=-2
CUT_PREFIX=0
WINDOW="8000 12000"     # ikki xil oyna uzunligi
```
Majburiy kirishlar: `MODEL_NAME_OR_PATH`, `DATA_DIR`, `OUTPUT_DIR`. `num2words` va `ctc_segmentation` paketlari kerak.

**`run_ctc_segmentation.py` argumentlari:** `--output_dir` (default `output`), `--data` (majburiy), `--window_len` (**default 8000**, "Window size for ctc segmentation algorithm"), `--sample_rate` (16000), `--model` (default `stt_en_fastconformer_ctc_large`), `--debug`, `--num_jobs` (-2). **Butun WAV fayl `wav.read()` bilan xotiraga o'qiladi** — chunklash yo'q; uzun audio uchun `window_len` lenta-oynasi ishlaydi. `index_duration = len(signal) / log_probs.shape[0] / sample_rate` sifatida hisoblanadi (kadr↔sekund konvertatsiyasi).

**`run_filter.sh` default thresholdlari (aynan) — sifat filtri qanday ko'rinishda bo'lishi kerakligining referensi:**
```
CER_THRESHOLD=30
WER_THRESHOLD=75
CER_EDGE_THRESHOLD=60
LEN_DIFF_RATIO_THRESHOLD=0.3
MIN_DURATION=1        # sekund
MAX_DURATION=20       # sekund
EDGE_LEN=5            # belgi soni
BATCH_SIZE=4
NUM_JOBS=-2
```
Ikki skript chaqiriladi: `transcribe_speech.py` (ASR bilan `pred_text` qo'shadi) va `get_metrics_and_filter.py` (metrikalarni hisoblab filtrlaydi).

> **Bu eng qimmatli topilma:** "moslashdan keyin ASR bilan qayta transkripsiya qilib, CER/WER bo'yicha filtrlash" — sanoat standarti. `CER_EDGE_THRESHOLD=60` + `EDGE_LEN=5` esa **segment chetlarida** (birinchi/oxirgi 5 belgi) xatolikni alohida tekshiradi — ya'ni "boshi kesilgan" segmentlarni tutish uchun.

---

### 2.9 WhisperX

**Manba:** https://github.com/m-bain/whisperX , https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py

1. **Kirish/chiqish.** CLI'da **ma'lum transkriptni moslash rejimi yo'q** — u transkripsiya + moslashni birga qiladi. Lekin **Python API'da bor**:
   ```python
   whisperx.align(
       transcript,               # Iterable[SingleSegment] — siz o'zingiz to'ldirishingiz mumkin
       model,                    # wav2vec2 align model
       align_model_metadata,
       audio,                    # path | np.ndarray | torch.Tensor
       device,
       interpolate_method="nearest",
       return_char_alignments=False,
       ...
   )
   ```
   Moslash **segment-ma-segment** ishlaydi: har segment uchun uning vaqt oralig'idagi audio kadrlarini olib, align modelini yugurtiradi va char/so'z vaqtlarini segment chegarasiga nisbatan hisoblaydi. Chiqish: so'z darajasidagi timestamp (+ ixtiyoriy char).

   **Muhim oqibat:** WhisperX'ga ma'lum transkriptni berish uchun sizga **avvaldan taxminiy segment vaqtlari kerak** — ya'ni u "toza" forced aligner emas, balki mavjud segmentlarni aniqlashtiruvchi. Soatlik audio + matn uchun o'zi yetarli emas.

2. **Uzun audio.** Whisper'ning 30 s oynalari + VAD asosida batch qilinadi; alignment esa segmentlar ichida.

3. **Til — o'zbek uchun BLOKER.** `DEFAULT_ALIGN_MODELS_TORCH`: en, fr, de, es, it. `DEFAULT_ALIGN_MODELS_HF`: ja, zh, nl, uk, pt, ar, cs, ru, pl, hu, fi, fa, el, tr, da, he, vi, ko, ur, te, hi, ca, ml, no (bokmål/nynorsk), sk, sl, hr, ro, eu, gl, ka, lv, fil, sv, id. **O'zbek yo'q.** Til topilmasa funksiya **xato loglaydi va `ValueError` ko'taradi**, HuggingFace'dan wav2vec2.0 model topib `--align_model` orqali berishni taklif qiladi.

   Ma'lum cheklov: *"Transcript words which do not contain characters in the alignment models dictionary e.g. '2014.' or '£13.60' cannot be aligned and therefore are not given a timing."* — ya'ni raqamlar/belgilar oldindan matn normalizatsiyasidan o'tishi shart.

4. **Aniqlik/tezlik.** *"70x realtime transcription using whisper large-v2"* (batched inference bilan). Alignment aniqligi uchun aniq ms raqam **ma'lumot topilmadi**.

5. **Windows.** `pip install whisperx`; ffmpeg va ba'zan rust kerak. Windows uchun alohida ogohlantirish yo'q.

6. **Litsenziya.** BSD-2-Clause.

7. **GPU.** `faster-whisper` backend bilan large-v2, `beam_size=5` uchun **<8GB VRAM**.

---

### 2.10 stable-ts

**Manba:** https://github.com/jianfch/stable-ts

1. **Kirish/chiqish.** **Ma'lum transkriptni moslashni to'g'ridan-to'g'ri qo'llaydi** — bu uning kuchli tomoni:
   ```python
   result = model.align('audio.mp3', text, language='en')
   ```
   ```bash
   stable-ts audio.mp3 --align text.txt --language en
   ```
   `--align` oddiy matn faylini yoki JSON natija faylini oladi. Chiqish: **so'z darajasidagi timestamp**. `align_words()` — har segmentni vaqt oralig'i bilan cheklab tezroq ishlaydi. Hujjat: *"when the text is correct but the timestamps need more work, `align()` is a faster alternative for testing various settings"*.

2. **Uzun audio.** Whisper ichidagi oynalash. `failure_threshold` parametri — *"abort alignment when percentage of words with zero duration exceeds"* shu qiymat. Ya'ni **buzilgan (nol davomiylikdagi so'zlar ko'p) alignmentni avtomatik to'xtatadi** — mismatch detektori sifatida foydali.

   `refine()` metodi: audio bo'laklarini ketma-ket o'chirib (mute qilib) token ehtimolliklarining o'zgarishini kuzatadi va har so'z uchun "eng kech boshlanish / eng erta tugash" vaqtini topib eng aniq chegarani aniqlaydi. Bu qimmat, lekin aniqlikni oshiradi.

3. **Til.** Whisper qo'llagan barcha tillar (Whisper o'zbek tilini zaif qo'llaydi — bu boshqa agent lanesi, lekin alignment uchun modelning shu tildagi sifatiga bog'liq).

4. **Aniqlik.** Aniq raqamlar **ma'lumot topilmadi**.

5. **Windows.** `pip install -U stable-ts`.

6. **Litsenziya.** MIT. ⚠️ **Repozitoriy 2026-05-30'da arxivlandi (read-only)** — yangi rivojlanish yo'q.

7. **GPU.** Whisper orqali CUDA.

---

### 2.11 whisper-timestamped

**Manba:** https://github.com/linto-ai/whisper-timestamped

1. **Kirish/chiqish.** ⚠️ **Ma'lum transkriptni moslash IMKONIYATI YO'Q** — u faqat o'zi transkripsiya qilib, keyin o'z chiqishini vaqtga bog'laydi. **Bizning masala uchun to'g'ridan-to'g'ri yaroqsiz.**
2. **Algoritm.** Whisper'ning **cross-attention** vaznlari ustida **DTW**. So'z moslash har segment dekodlangandan keyin "on the fly" bajariladi — qo'shimcha inference kerak emas.
3. **Til.** Whisper qo'llagan barcha tillar. Mualliflar wav2vec2 yondashuvining (WhisperX) kamchiligini ta'kidlaydi: har til uchun alohida model + qo'shimcha neyron tarmoq → masshtablash va xotira muammosi.
4. **Aniqlik.** Aniq foiz da'volari yo'q (**ma'lumot topilmadi**).
5. **Windows.** `pip3 install whisper-timestamped`, Python 3.7+ (3.9+ tavsiya), FFmpeg. Docker образlar ~9GB (GPU) / ~3.5GB (CPU).
6. **Litsenziya.** **AGPL-3.0** — tijorat uchun muammoli.
7. **GPU.** PyTorch orqali.

---

### 2.12 Gentle

**Manba:** https://github.com/lowerquality/gentle

1. **Kirish/chiqish.** Transkript majburiy (`transcript=@words.txt`). Chiqish JSON, so'z darajasida. (Gentle'ning mashhur xususiyati — har so'z uchun `case: "success"` yoki `"not-found-in-audio"` maydoni, bu mismatch aniqlash uchun juda qulay; lekin bu maydon nomlari men o'qigan sahifada aniq tasdiqlanmadi — **qism ma'lumot topilmadi**.)
2. **Uzun audio.** Kaldi asosida. Aniq chegaralar **ma'lumot topilmadi**.
3. **Til.** **Faqat ingliz tili.** O'zbek uchun yaroqsiz.
4. **Aniqlik.** Kürzinger va h.k. TEDlium v2 ustida: **0.41 s** o'rtacha chetlanish (HMM) — CTC-segmentation'ning 0.31 s'idan yomonroq, lekin aeneas'ning 9.01 s'idan ancha yaxshi. https://ar5iv.labs.arxiv.org/html/2007.09127
5. **Windows.** `./install.sh` faqat Mac/Linux. **Native Windows qo'llovi yo'q** → Docker (`lowerquality/gentle`).
6. **Litsenziya.** MIT.
7. **GPU.** Yo'q.

---

### 2.13 DSAlign (Mozilla) — mismatch bilan kurashish bo'yicha eng yaxshi algoritmik referens

**Manbalar:** https://github.com/mozilla/DSAlign , https://raw.githubusercontent.com/mozilla/DSAlign/master/doc/algo.md

1. **Kirish/chiqish.** Transkript majburiy. CLI:
   ```
   bin/align.sh --audio data/test1/audio.wav --script data/test1/transcript.txt \
       --aligned data/test1/aligned.json --tlog data/test1/transcript.log
   ```
   yoki `bin/align.sh --catalog data/test1.catalog`. Chiqish: `aligned.json` + `transcript.log`.

2. **Algoritm — bizning masala uchun eng muhim g'oya.** Yetti bosqichli pipeline:
   - **VAD** audioni uzluksiz nutq fragmentlariga bo'ladi;
   - har fragment DeepSpeech (STT) bilan transkripsiya qilinadi;
   - asl matn normalizatsiya qilinadi (kichik harf, punktuatsiya olib tashlanadi, bo'sh joy standartlashtiriladi) — STT chiqishiga mos shaklga keltiriladi;
   - **"recursive divide and conquer approach"**: global moslashga urinmaydi. Oralig'ning **o'rtasiga yaqin, uzun** iboralarga ustunlik beradigan tartiblangan ro'yxat yasaydi, keyin **Smith-Waterman** moslashni chuqurlikka bog'liq threshold'dan o'tgan ibora topilguncha ketma-ket qo'llaydi. Topilgan uzun moslik **anchor** bo'lib, chap va o'ng qismlarni rekursiv ravishda kichraytiradi.
   - Natija (aynan): *"Long non-matching chunks of spoken text or the original transcript will automatically and cleanly get ignored"* — uzun mos kelmaydigan bo'laklar avtomatik va toza tashlab yuboriladi; qisqa iboralar uzun mosliklar qo'ygan fazoviy cheklov ichida o'z-o'zidan joyiga tushadi.

   **Smith-Waterman ikki fazali:**
   - *Nomzod yasash:* asl matn qidiruv patterni uzunligidagi oynalarga bo'linadi, keyin **umumiy 3-gramma zichligi** bo'yicha reyting qilinadi (tez prefilter);
   - *Ballash:* match = **100**, mismatch = **−100**, gap = **−100** (default). Yakuniy ball maksimal belgi soniga bo'linib normallashtiriladi → maksimum ≈ 100.

3. **Sifat metrikalari va filtrlash.** `--output-<METRIC-ID>` metrika qiymatini natijaga qo'shadi, `--output-min-<METRIC-ID>` esa qabul threshold'ini belgilaydi. Metrikalar 0.0–1.0 oralig'ida (qo'llanadigan joyda).

4. **Aniqlik.** Aniq ms raqamlar **ma'lumot topilmadi**.
5. **Windows.** Virtualenv + DeepSpeech modeli; Windows haqida **ma'lumot topilmadi**. Amalda Linux.
6. **Litsenziya.** MPL-2.0.
7. **Holat.** DeepSpeech'ning o'zi to'xtatilgan → DSAlign'ni **ishlab chiqarishga tavsiya qilmaslik**, lekin **algoritmini ko'chirish** juda foydali.

---

### 2.14 audiogrep

**Manba:** https://github.com/antiboredom/audiogrep

Forced aligner **emas**. CMU Pocketsphinx bilan audioni transkripsiya qilib, keyin izlash so'zi bo'yicha "supercut" yasaydi. Transkript kerak emas (o'zi yasaydi). MIT. **2024-03'da arxivlangan**, muallif Videogrep'ga o'tishni tavsiya qiladi. **Bizning masala uchun yaroqsiz.**

---

### 2.15 ReadAlongs Studio — "akustik model yo'q" holati uchun qiziq alternativ

**Manba:** https://github.com/ReadAlongs/Studio

1. **Kirish/chiqish.** Transkript majburiy. Kirish: matn (plaintext yoki XML) + audio (WAV/MP3). Chiqish: ReadAlong XML, HTML, XHTML, **Praat TextGrid**, **ELAN .eaf**, **WebVTT**, **SRT** — so'z darajasida.
2. **Nega muhim.** Aynan **akustik modeli yo'q tillar** (Kanada mahalliy tillari) uchun yaratilgan: `g2p` mapping + **SoundSwallower** dekoder orqali *zero-shot* moslash qiladi — ya'ni yangi til uchun model o'qitish shart emas, faqat orfografiya→fonema qoidalari kerak. O'zbek lotin yozuvi juda muntazam bo'lgani uchun bunday g2p yozish oson.
3. **Uzun audio.** Aniq chegaralar **ma'lumot topilmadi**.
4. **Aniqlik.** Aniq raqamlar **ma'lumot topilmadi**.
5. **Windows.** `pip install readalongs`, Python 3.9+, FFmpeg, C kompilyator; *"Windows users can install via conda or manually"*.
6. **Litsenziya.** MIT (© 2019–2025 David Huggins-Daines va National Research Council Canada).
7. **GPU.** Yo'q (SoundSwallower yengil dekoder).

---

## 3. Juda uzun audio (soatlar): anchor asosidagi va rekursiv moslash

Bu masalaning klassik yechimi 1998-yilga borib taqaladi va zamonaviy vositalar ham shu g'oyaning variantlari.

### 3.1 Moreno va h.k. (1998) — rekursiv anchor algoritmi

**Manba:** Moreno, Joerg, Van Thong, Glickman, *"A Recursive Algorithm for the Forced Alignment of Very Long Audio Segments"*, ICSLP 1998, Sydney.
https://www.semanticscholar.org/paper/5aab9a55e3ced09929d3644b971085620cae22e4 , https://cris.biu.ac.il/en/publications/a-recursive-algorithm-for-the-forced-alignment-of-very-long-audio/

**G'oya:** forced alignment'ni **asta-sekin qisqarayotgan lug'at va til modeli bilan rekursiv nutq tanish** masalasiga aylantirish.
1. Birinchi o'tishda audioda **anchoring words** (tayanch so'zlar) topiladi.
2. Anchor = **N ta ketma-ket to'g'ri moslangan so'z** bo'lgan audio bo'lagi. N katta → aniqroq, N kichik → tezroq.
3. Anchor'larga nisbatan yozuv segmentlarga bo'linadi.
4. Ikkinchi o'tishda har segment ichida so'z vaqtlari aniqlanadi.

**Xususiyatlari:** akustik shovqinga va matn/audio bo'shliqlari va xatolariga **tolerant**. Sinov: 3 soatli TV/radio audiosi.

### 3.2 Zamonaviy amaliy variantlar (xotira muammosini qanday hal qilishadi)

| Yondashuv | Mexanizm | Manba |
|---|---|---|
| **Diagonal lenta (band) trellis** | To'liq O(M·N) trellis o'rniga faqat `[t − W/2, t + W/2]` oynasi → **O(M)**. Evristika: audio pozitsiyasi / matn pozitsiyasi nisbati deyarli doimiy. NeMo'da `--window_len` default **8000** (va `WINDOW="8000 12000"` bilan ikki marta yuguriladi) | https://ar5iv.labs.arxiv.org/html/2007.09127 , https://github.com/NVIDIA/NeMo/tree/main/tools/ctc_segmentation |
| **Sobit oynalar + overlap** | Audioni **30 s** oynalarga **2 s** kontekst overlap bilan bo'lib, emissionlarni ketma-ket hisoblash va ulash | `ctc-forced-aligner`: `--window_size 30 --context_size 2` — https://github.com/MahmoudAshraf97/ctc-forced-aligner |
| **`get_partitions()`** | ctc-segmentation paketining o'z funksiyasi: Transformer modellarida kvadratik murakkablikni oldini olish uchun audioni bo'laklaydi | https://github.com/lumaku/ctc-segmentation |
| **Rekursiv divide & conquer + anchor** | Oralig'ning o'rtasidagi uzun iborani Smith-Waterman bilan topib anchor qilish, keyin chap/o'ngni rekursiv qayta ishlash | DSAlign — https://raw.githubusercontent.com/mozilla/DSAlign/master/doc/algo.md |
| **VAD bilan oldindan bo'lish** | Avval Silero VAD/energiya bilan nutq bo'laklariga bo'lib, keyin har bo'lakni moslash | DSAlign, ManaTTS, arXiv:2510.03111 |
| **Kaldi cleanup recipe** | `steps/cleanup/segment_long_utterances.sh` (GMM) / `segment_long_utterances_nnet3.sh` (nnet3) — uzun yozuvlarni (TV shou, audiokitob) boshqarilar uzunlikdagi segmentlarga + mos transkript bo'lagiga bo'ladi; "light-supervised" sharoit uchun, ya'ni **transkript toza bo'lmasligi kutiladi**. Keyin `steps/cleanup/clean_and_segment_data.sh` mos kelmagan ma'lumotni yo'q qiladi | https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/cleanup/segment_long_utterances.sh , https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/cleanup/clean_and_segment_data.sh |
| **RAM bo'yicha sig'dirish** | DTW yondashuvida hamma narsa RAM'da: 4 GB ≈ 2 soat, 16 GB ≈ 10 soat | aeneas README |

**10 soatli audiokitob uchun amaliy retsept:** avval matnni bob/paragraf darajasida bo'lib, audioni ham **taxminiy** anchor'lar (masalan, bir necha noyob uzun ibora) bo'yicha ~10–30 daqiqalik bloklarga bo'ling; keyin har blokni CTC aligner'ga (30 s oyna + 2 s overlap bilan) bering. Bu ham xotirani, ham xato tarqalishini (drift) cheklaydi.

---

## 4. Algoritmik muammo A: matn va o'qilgan narsa mos kelmasa nima bo'ladi?

Real o'qish: so'z tashlab ketiladi, jumla takrorlanadi, xato o'qib qayta o'qiladi, "uh"/yo'tal qo'shiladi, kitobdagi sarlavha/izoh o'qilmaydi. Buni hal qiluvchi mexanizmlar (eng kuchlisidan boshlab):

### 4.1 `<star>` / wildcard token (eng to'g'ridan-to'g'ri yechim)
`torchaudio` MMS_FA modelida qo'shimcha emission o'lchovi — `<star>`. Vazifasi: *"Handles partial or mismatched transcripts by modeling unspecified content as wildcard tokens, improving alignment accuracy when transcript deletions exist."* Amalga oshirish: lug'atni kengaytirib, alignment'dan oldin emission'ga qo'shimcha o'lchov qo'shiladi. `get_model(with_star=False)` bilan o'chiriladi.
`ctc-forced-aligner`'da bu `--star_frequency` parametri: `edges` (default — faqat segment chetlarida) yoki `segment` (har segmentda).
Manbalar: https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html , https://docs.pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html , https://github.com/MahmoudAshraf97/ctc-forced-aligner

### 4.2 CTC-segmentation confidence (mismatch **detektori**)
Moslangan kadrlar L uzunlikdagi bo'laklarga bo'linadi, har bo'lakda o'rtacha ehtimollik, keyin **minimumi** olinadi (`score_min_mean_over_L`). Maqola: *"inflicts a penalty on the confidence score on mismatch, e.g., even if a single word is missing"*. **Tavsiya etilgan threshold: 0.22 (log fazoda −1.5)** dan past segmentlar tashlanadi. NeMo skriptida bu `MIN_SCORE=-2` sifatida yanada bo'shroq qo'yilgan.
Manbalar: https://github.com/lumaku/ctc-segmentation , https://ar5iv.labs.arxiv.org/html/2007.09127 , https://github.com/NVIDIA/NeMo/tree/main/tools/ctc_segmentation

### 4.3 Dinamik programmalash + fuzzy qidiruv (skip jarimasi bilan)
DSAlign: normalizatsiya → STT → **Smith-Waterman** (local alignment, match **100**, mismatch **−100**, gap **−100**) → 3-gramma zichligi bilan nomzod tanlash → **rekursiv anchor**. Natija: mos kelmagan uzun bo'laklar toza tashlab yuboriladi. Smith-Waterman aynan **local** moslash bo'lgani uchun tushib qolgan/qo'shilgan matnga tolerant (Needleman-Wunsch global moslashidan farqli).
Manba: https://raw.githubusercontent.com/mozilla/DSAlign/master/doc/algo.md

### 4.4 ASR bilan qayta transkripsiya + CER/WER filtri (sanoat standarti)
Moslashdan keyin **har segmentni ASR bilan qayta o'qitib**, `pred_text` ni `text` bilan solishtirish. NeMo `run_filter.sh` defaultlari: `CER_THRESHOLD=30`, `WER_THRESHOLD=75`, **`CER_EDGE_THRESHOLD=60` + `EDGE_LEN=5`** (chetlardagi 5 belgi alohida tekshiriladi — "boshi/oxiri kesilgan" segmentlarni tutish uchun), `LEN_DIFF_RATIO_THRESHOLD=0.3`.
Manba: https://raw.githubusercontent.com/NVIDIA/NeMo/main/tools/ctc_segmentation/run_filter.sh

ManaTTS (fors tili, arXiv:2409.07259, 2024-09-11) shu yondashuvni **CER bo'yicha darajalab** ishlatgan:
- **HIGH**: CER ≤ 0.05 → chunklarning **71.46%**
- **MIDDLE**: 0.05 < CER ≤ 0.2 → **28.54%**
- **REJECT**: CER > 0.2 → **2.02%** tashlangan
Natija: ~86 soat (44.1 kHz), 66,172 chunkdan **64,834** qabul qilindi; Tacotron2 MOS **3.76** (tabiiy nutq 4.01), 24,113 noyob so'z. (Foizlar manbada shunday keltirilgan — 71.46 + 28.54 = 100 bo'lib, 2.02% bilan kichik nomuvofiqlik bor; manbadagi ko'rinishda keltirildim.)
Manba: https://arxiv.org/html/2409.07259v1

**ManaTTS'ning mismatch yechimi** (o'z aligner'i, aeneas'ni *"audio-matn mos kelmasligiga sezgir"* deb rad etgan):
- bir nechta fors ASR modeli + **majority voting** bilan gipoteza transkript;
- **Interval Search** — belgilangan oraliqda aniq matn substringini topish;
- **Gapped Search** — aniq moslik bo'lmasa, **ruxsat etilgan bo'shliqlar bilan** topish.
Mismatch sabablari deb sanaganlari: speaker variatsiyalari, senzura, o'z-o'zini tuzatish (self-correction), transkripsiya xatolari.

### 4.5 Nol davomiylikdagi so'zlar bo'yicha abort
stable-ts `failure_threshold`: *"abort alignment when percentage of words with zero duration exceeds"* shu qiymat. Arzon va samarali "alignment buzildi" signali.
Manba: https://github.com/jianfch/stable-ts

### 4.6 Matn tozalash + normalizatsiya (majburiy oldingi qadam)
- NeMo: `USE_NEMO_NORMALIZATION='True'`, `num2words` paketi (raqamlarni so'zga aylantirish).
- WhisperX cheklovi: *"Transcript words which do not contain characters in the alignment models dictionary e.g. '2014.' or '£13.60' cannot be aligned"* → raqamlar/valyuta/qisqartmalar oldindan yozib chiqilishi shart.
- MMS/torchaudio: `normalize_uroman()` — kichik harf, `[^a-z' ]` → bo'sh joy, ketma-ket bo'sh joylarni siqish.
- LJSpeech'ning o'zi ham "Normalized Transcription" ustuniga ega — raqam, tartib son, pul birliklari to'liq so'z bilan yozilgan.

### 4.7 Biased LM / lightly supervised dekodlash (Kaldi yo'li)
`steps/cleanup/segment_long_utterances.sh` "light-supervised training scenario" uchun mo'ljallangan, *"where the input transcription is not expected to be completely clean and may have significant errors"* — berilgan matndan biased til modeli yasab dekodlaydi, shuning uchun o'qilmagan/qo'shilgan matnni yutadi.
Manba: https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/cleanup/segment_long_utterances.sh

---

## 5. Algoritmik muammo B: qayerdan kesish? (2–12 s, tabiiy pauza, so'z o'rtasidan emas)

Bu **uchta signalning kombinatsiyasi**: (1) matn chegaralari, (2) alignment timestamp'lari, (3) VAD/sukunat.

### 5.1 Matn tomonidan nomzod chegaralari
- **NFA:** `additional_segment_grouping_separator`, **default `['.', '?', '!', '...']`** → segment = jumla. https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html
- **NeMo CTC-Segmentation tool:** `ADDITIONAL_SPLIT_SYMBOLS=":|;"` (nuqta/savol/undov ustiga qo'shimcha) + `MAX_SEGMENT_LEN=30` — juda uzun jumlalarni majburan kichraytirish uchun. https://raw.githubusercontent.com/NVIDIA/NeMo/main/tools/ctc_segmentation/run_segmentation.sh
- **MMS data_prep:** matn faylining **har satri** = bitta kerakli segment → bo'lish qarori butunlay sizning matn preprocessingingizda. https://github.com/facebookresearch/fairseq/tree/main/examples/mms/data_prep
- **LibriTTS:** kitob matni paragraflarga, paragraflar jumlalarga bo'lingan va nutq **"at sentence breaks"** kesilgan. Eslatma: LibriSpeech'ning o'zi **0.3 s dan uzun sukunatlar** bo'yicha kesgan edi, LibriTTS buni jumla chegarasiga o'zgartirdi. https://ar5iv.labs.arxiv.org/html/1904.02882

### 5.2 Alignment timestamp'lari — so'z o'rtasidan kesmaslikning kafolati
So'z darajasidagi `start`/`end` bo'lgani uchun kesish nuqtasi **hech qachon so'z ichiga tushmaydi** — siz faqat so'zlar orasidagi bo'shliqda kesasiz. Vaqt rezolyutsiyasi: MMS'da `stride_msec = 20` → **50 kadr/sekund** → ±20 ms.

**MMS `get_spans()` ning nozik qoidasi:** ikki segment orasidagi pauza **ikkiga bo'linadi**:
```python
pad_start = prev_seg.start if (idx == 0) else int((prev_seg.start + prev_seg.end)/2)
```
Ya'ni har segment o'zidan oldingi jimlikning yarmini oladi. Bu TTS uchun juda muhim — segment boshidagi "nafas"/attack yo'qolmaydi va oxirida tabiiy tugash qoladi.
Manba: https://raw.githubusercontent.com/facebookresearch/fairseq/main/examples/mms/data_prep/align_utils.py

> **CTC "peaky behavior" ogohlantirishi:** torchaudio tutoriali aniq yozadi — CTC modellar label paydo bo'lganda spike beradi, lekin labelning **to'liq davomiyligini qoplamaydi**, va blank token talqini kontekstga qarab o'zgaradi → **so'z chegarasi timestamp'i o'z-o'zidan aniq emas**. Shu sababli faqat alignment timestamp'ga tayanmaslik, VAD bilan "snap" qilish kerak.

### 5.3 VAD bilan haqiqiy sukunatga "snap" qilish
**Silero VAD** (https://github.com/snakers4/silero-vad , https://pytorch.org/hub/snakers4_silero-vad_vad/):
- `get_speech_timestamps(wav, model, return_seconds=True)`;
- sample rate: **8000 Hz va 16000 Hz**;
- tezlik: *"One audio chunk (30+ ms) takes less than 1ms to be processed on a single CPU thread"*; ONNX bilan 4–5x tezroq;
- model hajmi ~**2 MB** (JIT);
- 6000+ tilni qamrovchi korpusda o'qitilgan → **o'zbek uchun ham ishlaydi**;
- **Litsenziya: MIT**, telemetriya/kalit/registratsiya yo'q.

**Muhim xatti-harakat:** *"Chunks longer than `max_speech_duration_s` will be split at the timestamp of the last silence that lasts more than 100ms to prevent aggressive cutting; otherwise, they will be split aggressively just before `max_speech_duration_s`"* — ya'ni Silero'ning o'zi allaqachon "oxirgi 100 ms dan uzun sukunatda kes" qoidasini qo'llaydi. Bizga aynan shu qoida kerak.

Asosiy parametrlar: `threshold`, `min_speech_duration_ms`, `max_speech_duration_s`, `min_silence_duration_ms`, `speech_pad_ms` — aniq default qiymatlari men o'qigan sahifada ko'rsatilmagan (**ma'lumot topilmadi**; kodda `utils_vad.py` da).

### 5.4 2–12 sekund oynasi — real qiymatlar

| Manba | Min | Max | Izoh |
|---|---|---|---|
| **LJSpeech** (haqiqiy dataset) | **1.11 s** | **10.10 s** | 13,100 klip, o'rtacha **6.57 s**, jami 23:55:17. *"The clips were segmented automatically based on silences in the recording, with boundaries generally aligning with sentence or clause boundaries"*. 22050 Hz, 16-bit PCM, mono |
| **ManaTTS** (retsept) | **2 s** | **12 s** | *"The audio is divided into smaller parts using silent intervals. We ensure these audio parts are between 2 and 12 seconds"* — kichik bo'laklarni **birlashtirish** yoki **sukunat aniqlash sozlamasini o'zgartirish** orqali |
| **NeMo CTC-Seg filter** | `MIN_DURATION=1` | `MAX_DURATION=20` | `MAX_SEGMENT_LEN=30` (matn bo'lishda) |
| **ctc-forced-aligner** | — | — | `--merge_threshold` bilan qo'shni segmentlarni birlashtirish |

Manbalar: https://www.kaggle.com/datasets/awsaf49/ljspeech-dataset , https://arxiv.org/html/2409.07259v1 , https://raw.githubusercontent.com/NVIDIA/NeMo/main/tools/ctc_segmentation/run_filter.sh

**Amaliy algoritm (yuqoridagilarni birlashtirib):**
1. Matnni jumlalarga bo'l (`. ? ! …`), keyin juda uzunlarini `; : ,` bo'yicha qo'shimcha bo'l.
2. Forced alignment bilan har jumlaning `start`/`end` ini ol (so'z darajasida).
3. Silero VAD'dan sukunat oraliqlari ro'yxatini ol.
4. Har chegara uchun: alignment chegarasidan ±300 ms atrofdagi **eng yaqin sukunat markazi**ga "snap" qil. Sukunat topilmasa (so'z o'rtasi) — bu chegarani **bekor qil** va qo'shni segment bilan birlashtir.
5. Uzunlik < 2 s → keyingi segment bilan birlashtir (agar natija ≤ 12 s bo'lsa).
6. Uzunlik > 12 s → ichidagi eng uzun sukunatda (yoki `; :` bo'yicha) qayta bo'l.
7. Har segmentga 100–200 ms sukunat "padding" qo'sh (MMS'ning yarim-pauza qoidasi bo'yicha).
8. LibriTTS uslubidagi qo'shimcha evristikalar: juda ko'p so'zli "jumla" (LibriTTS **71 so'z**dan ortiqni tashlab yuborgan — jumla bo'lish xatosi belgisi) va **katta o'rtacha so'z davomiyligi** (audio/matn mos kelmasligi belgisi) bo'lgan segmentlarni tashla.
9. ASR bilan qayta transkripsiya → CER/WER filtri (4.4).

---

## 6. Standart chiqish formati va pipeline shakli

### 6.1 Konvergensiya bo'lgan pipeline shakli

```
1. Matn tayyorlash
   - normalizatsiya (raqam→so'z, punktuatsiya, kichik harf variant)
   - jumlalarga bo'lish  → har satr = nomzod segment
2. Audio tayyorlash
   - 16 kHz mono (alignment uchun) + asl sifat (TTS uchun saqlanadi)
   - juda uzun fayl bo'lsa: anchor/blok bo'lish
3. Forced alignment (CTC)
   - emission (30 s oyna + 2 s overlap) → trellis (diagonal lenta) → backtrack
   - chiqish: so'z/belgi darajasidagi (start, end, score)
4. Segmentlash
   - matn chegarasi + alignment timestamp + VAD sukunatiga snap
   - 2–12 s oynaga sig'dirish (birlashtirish / qayta bo'lish)
5. Sifat filtri   [boshqa agent lanesi, lekin shakl shu]
   - alignment confidence (< 0.22 / log −1.5 tashla)
   - ASR qayta transkripsiya → CER/WER/edge-CER
   - davomiylik, SNR
6. Eksport
   - wav/ ga kesish + metadata.csv (LJSpeech) yoki manifest.json (NeMo/MMS)
```

### 6.2 Format variantlari

**LJSpeech** (bizning maqsad formati) — https://www.kaggle.com/datasets/awsaf49/ljspeech-dataset :
- `wavs/LJ001-0001.wav` — **22050 Hz, 16-bit PCM, mono**
- `metadata.csv` — **`|` (0x7c) bilan ajratilgan**, har satr bitta yozuv. Men o'qigan manba maydonlar sifatida **ID** (mos `.wav` fayl nomi) va **Normalized Transcription** (raqamlar, tartib sonlar va pul birliklari to'liq so'zga yozilgan, UTF-8) ni sanaydi. ⚠️ Amaliyotda ko'p TTS trainerlari **uch ustunli** (`ID|transcription|normalized_transcription`) variantni kutadi — bu men o'qigan sahifada tasdiqlanmadi, **alohida tekshirish kerak**.

**NeMo/MMS manifest** (JSONL, har satr bitta segment):
```json
{"audio_filepath": "...", "duration": 5.12, "text": "...", "normalized_text": "...", "uroman_tokens": "...", "audio_start_sec": 123.44}
```
Manba: https://github.com/facebookresearch/fairseq/tree/main/examples/mms/data_prep

**CTM** (NFA) — `<utt_id> 1 <start_seconds> <duration_seconds> <text>`, token/word/segment darajalarida alohida fayl.

**TextGrid** (MFA, ReadAlongs) — Praat formati, so'z **va** fonema qatlamlari. Fonema darajasidagi datasetlar (masalan FastSpeech2 duration predictor) uchun kerak.

**Oraliq saqlash uchun tavsiya:** so'z darajasidagi to'liq alignment'ni **bir marta** hisoblab JSON'ga saqlang (`{word, start, end, score}`), keyin segmentlash/filtrlash qoidalarini shu JSON ustida qayta-qayta yugurtirsangiz, har eksperimentda GPU ishini takrorlamaysiz. (Bu men ko'rgan vositalarning hech birida majburiy emas — amaliy tavsiya.)

---

## 7. Windows 11 + GPU + o'zbek tili uchun tavsiya

### Asosiy tavsiya: `ctc-forced-aligner` (MMS-300m) + Silero VAD

**Nega:**
1. **O'zbek akustik modeli kerak emas** — MMS-300m 1130+ tilda o'qitilgan, char-level CTC + romanizatsiya. O'zbek lotin alifbosi `a-z` + `'` ga deyarli mos → `--language uzb --romanize` (yoki romanizatsiyasiz) ishlaydi. MFA yo'lida esa avval talaffuz lug'ati yasab, keyin akustik model o'qitish kerak — va uni o'qitish uchun **allaqachon segmentlangan** dataset kerak (tovuq-tuxum).
2. **Uzun audio ichida hal qilingan** — `--window_size 30 --context_size 2`, torchaudio'ning xom `forced_align`'idan **5x kam xotira**, `batch_size==1` cheklovi o'ralgan.
3. **Windows'da og'riq yo'q** — `pip install` + FFMPEG. Kaldi yo'q, conda muhit jangi yo'q, Docker yo'q, WSL yo'q. Taqqoslash uchun: MFA 3.x Windows holati noaniq (Docker/WSL ehtimoli yuqori), NeMo Linux/Docker'ga yo'naltirilgan, Gentle native Windows'ni qo'llamaydi, aeneas espeak linking xatolari bilan mashhur.
4. **GPU'ni ishlatadi** (`--device cuda`).
5. **Mismatch'ga chidamli** — `--star_frequency` (wildcard token) + CTC confidence.
6. **Granularlik tanlanadi** — `--split_size sentence|word|char`; so'z darajasi bizga kesish nuqtalarini beradi.
7. **Litsenziya:** kod BSD-2 (toza). ⚠️ **Default model CC-BY-NC 4.0** — agar dataset/model tijoriy bo'lsa, `--alignment_model` orqali Apache/MIT litsenziyali wav2vec2 XLSR modelga o'ting (o'zbek uchun fine-tune qilingan modellar bor-yo'qligi boshqa agent lanesi).

**Zaxira / qo'shimcha qatlamlar:**
- **Sifat nazorati:** NeMo `run_filter.sh` mantiqini ko'chirib yozing (ASR qayta transkripsiya → CER 30 / WER 75 / edge-CER 60 / len-diff 0.3) — vositaning o'zini o'rnatmasdan, faqat mantiqni.
- **Mismatch og'ir bo'lsa:** DSAlign'ning rekursiv anchor + Smith-Waterman (match 100 / mismatch −100 / gap −100, 3-gramma prefilter) algoritmini qo'lda implement qiling — DSAlign'ning o'zini emas (DeepSpeech o'lgan).
- **Fonema darajasi kerak bo'lsa (FastSpeech2 kabi):** birinchi bosqichda MMS bilan segmentlagandan **keyin**, qisqa segmentlar ustida `mfa train` bilan o'zbek akustik modelini o'qitib, `mfa align` bilan fonema TextGrid oling. Windows'da MFA'ni Docker orqali ishlatishga tayyor bo'ling.
- **Tez prototip:** `stable-ts --align text.txt` — MIT, oson, `failure_threshold` bor; lekin repozitoriy arxivlangan va Whisper'ning o'zbek sifatiga bog'liq.
- **Qat'iy tekshirish:** `ctc-segmentation` paketini to'g'ridan-to'g'ri (Apache-2.0, aniqligi o'lchangan: 0.31 s o'rtacha chetlanish, 88.8% ±0.5 s ichida) o'z wav2vec2 modelingiz bilan ishlatish — eng nazorat ostidagi, lekin ko'proq kod yozish kerak.

### Tavsiya etilmaydiganlar (bu masala uchun)
- **whisper-timestamped** — ma'lum transkriptni moslay olmaydi + AGPL-3.0.
- **audiogrep** — forced aligner emas, arxivlangan.
- **Gentle** — faqat ingliz, native Windows yo'q.
- **aeneas** — UZB ro'yxatda yo'q, Windows C-extension og'rig'i, TEDlium'da 9.01 s chetlanish, AGPL-3.0.
- **WhisperX** (yakka holda) — o'zbek align modeli yo'q → `ValueError`; ma'lum transkript uchun oldindan segment vaqtlari kerak.
- **MFA birinchi qadam sifatida** — uzun uzluksiz audioni bo'lish uchun mo'ljallanmagan.

---

## 8. "Ma'lumot topilmadi" ro'yxati (halol inventar)

- MFA 3.x uchun **native Windows** qo'llovi: rasmiy hujjatda aniq bayonot yo'q; `kalpy` win-64 build mavjudligi tasdiqlanmadi. MFA 2.1.7 hujjati bilan MFA 3.x hujjati bu masalada **ziddiyatli** ko'rinadi (2.1.7: "all features on all platforms" 2.0.6'dan keyin; 3.x: Windows faqat Docker konteksida eslatiladi).
- MFA uchun **aniq alignment xatolik (ms)** raqamlari.
- MFA akustik model o'qitish uchun **minimal soat** talabi; monophone/triphone/LDA/SAT bosqichlari ro'yxati.
- **RTF (realtime factor)** raqamlari: `ctc-forced-aligner`, NFA, torchaudio `forced_align`, ctc-segmentation, aeneas uchun rasmiy tezlik o'lchovlari topilmadi. Faqat WhisperX'ning **70x realtime** transkripsiya raqami (alignment'ning emas) va Silero VAD'ning **<1 ms / 30 ms chunk** raqami bor.
- **MMS_FA aniqligi** (ms xatolik) — rasmiy raqam yo'q.
- **Gentle'ning JSON maydon nomlari** (`case: "not-found-in-audio"` kabi) rasmiy sahifada tasdiqlanmadi.
- **Silero VAD `get_speech_timestamps` default qiymatlari** (threshold, min_speech_duration_ms, max_speech_duration_s, min_silence_duration_ms, speech_pad_ms) — README sahifasida ko'rsatilmagan.
- **DSAlign** uchun Windows qo'llovi va aniq metrika ID ro'yxati.
- **ReadAlongs** uzun audio chegaralari va aniqlik raqamlari.
- **LJSpeech `metadata.csv`** ustunlar soni (2 yoki 3) — o'qigan manba 2 ta maydonni sanadi, amaliyotda 3 ustunli variant keng tarqalgan; tekshirish kerak.
- **aeneas** eSpeak-ng orqali `uz` ovozini qo'shib o'zbek tilini qo'llash imkoniyati.
- **arXiv:2510.03111** ("Evaluation of preprocessing pipelines in the creation of in-the-wild TTS datasets", 2025-10-06) — PDF'dan aniq raqamli thresholdlar va pipeline solishtirish natijalari chiqarib olinmadi; faqat Silero VAD baseline sifatida, MFA va Whisper ishlatilgani, hamda PESQ/MOS/F0 metrikalari bilan baholanishi aniqlandi.
- **ctc-segmentation** uchun Gentle/aeneas'dan tashqari aligner'lar bilan solishtirish va std/±0.5 s raqamlari (faqat CTC-segmentation uchun to'liq: 0.31 s / 0.85 s / 88.8%).

---

## 9. Barcha manbalar

**Rasmiy hujjat / repozitoriy**
- aeneas: https://github.com/readbeyond/aeneas , https://www.readbeyond.it/aeneas/docs/ , https://github.com/readbeyond/aeneas/blob/master/wiki/INSTALL.md
- aeneas Windows muammolari: https://github.com/readbeyond/aeneas/issues/306 , https://github.com/ReadBeyond/aeneas/issues/315 , https://github.com/sillsdev/aeneas-installer/releases
- MFA: https://montreal-forced-aligner.readthedocs.io/ , https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html , https://montreal-forced-aligner.readthedocs.io/en/v2.1.7/installation.html , https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/train_acoustic_model.html , https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner , https://raw.githubusercontent.com/MontrealCorpusTools/Montreal-Forced-Aligner/main/docs/source/installation.rst , https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/270 , https://anaconda.org/conda-forge/montreal-forced-aligner
- MFA modellari: https://mfa-models.readthedocs.io/en/latest/acoustic/index.html
- ctc-segmentation: https://github.com/lumaku/ctc-segmentation
- ctc-segmentation maqolasi: https://arxiv.org/abs/2007.09127 , https://ar5iv.labs.arxiv.org/html/2007.09127
- torchaudio: https://docs.pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html , https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html , https://docs.pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html , https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.MMS_FA.html
- ctc-forced-aligner: https://github.com/MahmoudAshraf97/ctc-forced-aligner , https://pypi.org/project/ctc-forced-aligner/ , https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner
- MMS data_prep: https://github.com/facebookresearch/fairseq/tree/main/examples/mms/data_prep , https://raw.githubusercontent.com/facebookresearch/fairseq/main/examples/mms/data_prep/align_utils.py
- NFA: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html , https://github.com/NVIDIA/NeMo
- NeMo CTC-Segmentation tool: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/ctc_segmentation.html , https://github.com/NVIDIA/NeMo/tree/main/tools/ctc_segmentation , https://raw.githubusercontent.com/NVIDIA/NeMo/main/tools/ctc_segmentation/run_segmentation.sh , https://raw.githubusercontent.com/NVIDIA/NeMo/main/tools/ctc_segmentation/run_filter.sh , https://raw.githubusercontent.com/NVIDIA/NeMo/main/tools/ctc_segmentation/scripts/run_ctc_segmentation.py
- WhisperX: https://github.com/m-bain/whisperX , https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
- stable-ts: https://github.com/jianfch/stable-ts
- whisper-timestamped: https://github.com/linto-ai/whisper-timestamped
- Gentle: https://github.com/lowerquality/gentle
- DSAlign: https://github.com/mozilla/DSAlign , https://raw.githubusercontent.com/mozilla/DSAlign/master/doc/algo.md
- audiogrep: https://github.com/antiboredom/audiogrep
- ReadAlongs Studio: https://github.com/ReadAlongs/Studio
- Silero VAD: https://github.com/snakers4/silero-vad , https://pytorch.org/hub/snakers4_silero-vad_vad/ , https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
- Kaldi cleanup: https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/cleanup/segment_long_utterances.sh , https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/cleanup/clean_and_segment_data.sh , https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/cleanup/segment_long_utterances_nnet3.sh

**Maqolalar**
- Kürzinger, Winkelbauer, Li, Watzel, Rigoll (2020), "CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition", Speech and Computer, Springer, 267–278. arXiv:2007.09127 (v1 2020-07-17, v2 2020-10-05)
- Moreno, Joerg, Van Thong, Glickman (1998), "A Recursive Algorithm for the Forced Alignment of Very Long Audio Segments", ICSLP 1998: https://www.semanticscholar.org/paper/5aab9a55e3ced09929d3644b971085620cae22e4 , https://cris.biu.ac.il/en/publications/a-recursive-algorithm-for-the-forced-alignment-of-very-long-audio/
- Zen va h.k. (2019), "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech", arXiv:1904.02882: https://ar5iv.labs.arxiv.org/html/1904.02882
- ManaTTS Persian (2024-09-11), arXiv:2409.07259: https://arxiv.org/html/2409.07259v1
- "Evaluation of preprocessing pipelines in the creation of in-the-wild TTS datasets" (2025-10-06), arXiv:2510.03111: https://arxiv.org/pdf/2510.03111
- Pratap va h.k. (2023), "Scaling Speech Technology to 1,000+ Languages" (MMS) — torchaudio MMS_FA hujjati orqali
- McAuliffe, Gunter, Wagner, Sonderegger (2026), "Montreal Forced Aligner and the state of speech-to-text alignment in 2026", Interspeech 2026

**Dataset formatlari**
- LJSpeech: https://www.kaggle.com/datasets/awsaf49/ljspeech-dataset , https://huggingface.co/datasets/flexthink/ljspeech

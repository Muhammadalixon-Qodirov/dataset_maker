# Tadqiqot 5: Mavjud vositalar, workflow va UX amaliyoti

> **Lane**: Nima allaqachon mavjud (qayta ixtiro qilmaslik uchun) va dataset yaratish workflow/UX amalda qanday bajariladi.
> **Lane emas**: forced-alignment algoritm ichki mexanikasi, korpus tarixi, o'zbek tili xususiyatlari, sifat metrikasi chegaralari — bularni boshqa agentlar qamrab oladi.
> **Sana**: 2026-09-29. Barcha yulduz/commit ma'lumotlari GitHub API va repo sahifalaridan shu kunga olingan.

---

## 0. Eng muhim xulosa (qisqa)

Foydalanuvchining keysi — **to'liq matn + uzun audio → LJSpeech dataset** — bozorda **to'liq tayyor, pishiq, Windows'da ishlaydigan yechim sifatida MAVJUD EMAS**. Mavjud vositalar ikki qutbga bo'linadi:

1. **Whisper-asosli "matn kerak emas" generatorlar** (ozen-toolkit, speech-dataset-generator, tts-dataset-generator...) — ular foydalanuvchining eng katta boyligini, ya'ni **haqiqiy ground-truth matnni butunlay tashlab yuboradi** va o'rniga Whisper xatolarini datasetga yozadi.
2. **Audiokitob hizalash vositalari** (easyaligner, afaligner, syncabook, Storyteller, timething, aeneas) — ular matnni ishlatadi, lekin chiqishi EPUB3/SMIL/JSON sync-map, **LJSpeech emas**; ko'plari Windows'da og'riqli yoki tashlangan.

Ya'ni bo'shliq aynan shu ikkisining **oralig'ida**: *hizalash natijasini TTS-sifatli segmentlarga aylantirish + inson ko'rigi UI*. Aynan shu qismni yozish kerak, qolgani qayta ishlatiladi.

---

## 1. TO'G'RIDAN-TO'G'RI KEYS: kitob matni + uzun audio hizalash

Bu bo'lim foydalanuvchining vazifasiga eng yaqin vositalar. Diqqat bilan o'qilsin.

### 1.1 youmebangbang/TTS-dataset-tools — keysga eng aniq mos keluvchi mavjud vosita
- URL: https://github.com/youmebangbang/TTS-dataset-tools
- MIT, 52 yulduz, 17 fork
- **Kirish**: uzun audio + unga tegishli **to'liq matn fayli**. **Chiqish**: kesilgan segmentlar + matn, proofreading uchun tahrirlanadigan ko'rinishda.
- Ikki rejim: (a) **aeneas** bilan matnni audioga force-align qilish, (b) Google Speech-to-Text API bilan diarization + transkripsiya.
- Maksimal sukunat oralig'i bo'yicha kesadi, `custom max length` qo'llab-quvvatlanadi.
- **KRITIK zaiflik (Windows)**: README'da aniq yozilgan — *"Linux environment is recommended for Aeneas option, in windows aeneas will not be able to make longer cuts due to memory issues"*. Ya'ni Windows'da aeneas uzun kesimlarni qila olmaydi.
- Boshqa zaifliklar: proofreading UI'da kolonka kengligini qo'lda sozlash kerak; matn maydonlari navigatsiyada yangilanishi uchun fokusni almashtirish kerak; VPN Google API'ni buzadi.
- Oxirgi commit sanasi: **ma'lumot topilmadi** (repo sahifasida ko'rinmadi).
- **Xulosa**: g'oyasi to'g'ri, bajarilishi eskirgan va Windows'da sinmaydi. Ilhom sifatida o'qish kerak, asos sifatida emas.

### 1.2 kb-labb/easyaligner — TEXNIK JIHATDAN ENG YAXSHI TANLOV
- URL: https://github.com/kb-labb/easyaligner · Blog: https://huggingface.co/blog/KBLab/easyaligner
- MIT, 24 yulduz, yaratilgan 2025-06-24, oxirgi push **2026-07-04** (tirik, lekin yosh loyiha)
- Shvetsiya Qirollik kutubxonasi (KBLab) tomonidan. RixVox-v2 korpusi (~23 000 soat nutq) aynan shu bilan yaratilgan — ya'ni real production sinovidan o'tgan.
- **Uzun audio**: to'g'ridan-to'g'ri bloggan — *"It can be used to align hours of audio in a single pass — without the need to segment the audio into smaller chunks"*. Bu 10 soatli audio uchun aynan kerakli xususiyat: bo'lak chegaralarida xato paydo bo'lmaydi.
- Pipeline: **VAD** (pyannote — gated, HF token kerak; yoki **silero** — token kerak emas) → **wav2vec2 emissions** → PyTorch'ning GPU-tezlashtirilgan forced-alignment API (Viterbi).
- **Kirish**: audio (MP3 va h.k.) + matn (string yoki char-span indeksli strukturalangan segmentlar). **Chiqish**: JSON — segment start/end + so'z darajasidagi timestamp **va confidence ball**.
- Matn-audio mos kelmasligi uchun 3 ta rejim hujjatlashtirilgan: (1) to'liq qoplash, (2) qismli qoplash + ma'lum audio chegaralari, (3) qismli qoplash + noma'lum chegaralar → **ASR chiqishiga fuzzy matching bilan kerakli sohani topish**. Bu audiokitobdagi disclaimer/intro/outro muammosini hal qiladi.
- Ko'p tilli holat uchun **Meta MMS-1B-all** tavsiya qiladi → o'zbek uchun ham yo'l bor.
- O'rnatish: `pip install easyaligner --extra-index-url https://download.pytorch.org/whl/cu128`
- Yondosh vositalar: `easytranscriber` (ASR), `easysearch` (interaktiv ko'rish).
- **Zaiflik**: 24 yulduz — jamoa kichik, Windows aniq test qilinganligi hujjatlashtirilmagan (lekin sof PyTorch + pip bo'lgani uchun Windows'da ishlashi kutiladi). Aniq benchmark raqamlari yo'q. Failure mode'lar hujjatlashtirilmagan.
- **Confidence ball chiqarishi** review triage uchun oltin qadrli: past ballli segmentlarni birinchi ko'rsatish mumkin.

### 1.3 jianfch/stable-ts — API jihatidan eng qulay, lekin ARXIVLANGAN
- URL: https://github.com/jianfch/stable-ts
- MIT, 2281 yulduz, **2026-05-30 da ARXIVLANGAN (read-only)**, 0 ochiq issue
- `align()` — mavjud to'liq matnni (plain text yoki tokenlar) audioga so'z darajasida hizalaydi; transkripsiyadan **ancha tez**. `WhisperResult` obyektini qayta uzatib, inference'ni qaytadan yugurtirmasdan boshqa sozlamalar bilan qayta hizalash mumkin.
- `align_words()` — har segmentni timestamp oralig'i bilan cheklaydi; `align()`dan **keskin tezroq**, segment chegaralari aniq bo'lganda so'z-timestamp xatosini kamaytiradi.
- `refine()` — audio qismlarini ketma-ket ovozsizlantirib (mute) token ehtimolliklarini kuzatadi va har so'zning "eng kech boshlanishi / eng erta tugashi"ni topadi. Bu **so'z o'rtasida kesilishga qarshi** eng nozik mexanizm.
- `regroup()` — zanjirlanadigan metodlar: gap bo'yicha bo'lish, punktuatsiya bo'yicha bo'lish, merge, uzunlik bo'yicha bo'lish. **Operatsiyalarni string sifatida yozib, reproducible qilib saqlash mumkin** — bu arxitektura uchun juda yaxshi pattern (segmentatsiya qoidasi = konfiguratsiya stringi).
- `nonspeech_skip` (berilgan davomdan uzun sukunatlarni o'tkazib yuborish), `max_instant_words` (nol-davomli so'zlar ko'p bo'lgan segmentni tashlash).
- Chiqish: SRT / VTT / ASS / TSV / JSON — so'z va segment darajasida bir vaqtda.
- **Zaiflik**: arxivlangan → bug fix bo'lmaydi, yangi PyTorch/Whisper versiyalari bilan mos kelmasligi vaqt masalasi. Lekin MIT — fork qilib olish mumkin va kod hajmi boshqarilishi mumkin.

### 1.4 r4victor/afaligner — WINDOWS'DA ISHLAMAYDI
- URL: https://github.com/r4victor/afaligner
- MIT, 105 yulduz, oxirgi push 2025-08-09, 5 ochiq issue
- DTW variantini (FastDTW, chiziqli vaqt/xotira) ishlatadi: matnni **sintez qiladi** (eSpeak) va sintez qilingan audioni yozib olingan audio bilan hizalaydi.
- Chiqish: JSON yoki SMIL sync map (`begin`/`end` timestamp'lar, masalan `"0:00:02.600"`).
- Audiokitoblardagi boshlanish/tugash strukturasi farqini (disclaimer) ataylab hisobga oladi.
- Talab: aeneas, numpy, jinja2, FFmpeg, eSpeak.
- **README'da aniq**: *"Windows is not currently supported (you may try to use a VM)."* → foydalanuvchi uchun bu deyarli to'xtatuvchi omil (WSL kerak).
- Fork'lar mavjud (FranPregernik/afaligner) lekin ular ham bir xil cheklovga ega.

### 1.5 r4victor/syncabook
- URL: https://github.com/r4victor/syncabook
- MIT, 360 yulduz, 74 commit
- Kirish: matn (plaintext yoki XHTML) + audio (.wav/.mp3). **Chiqish: EPUB3 + Media Overlays** — ya'ni "read-along" kitob. **TTS dataset emas.**
- afaligner ustiga qurilgan → afaligner'ning Windows cheklovini meros oladi. Docker alternativi bor.
- Matnni audio segmentlariga mos ravishda oldindan bo'lish sifatni oshiradi (ya'ni matn tayyorlash bosqichi baribir zarur).
- **Foydasi**: matn bo'linishi + hizalash + tekshirish oqimini qanday tashkil qilgani o'rganishga arziydi; kod sifatida qayta ishlatilmaydi.

### 1.6 smoores-dev/storyteller — production servis arxitekturasi namunasi
- Mirror: https://github.com/smoores-dev/storyteller · Kanonik: https://gitlab.com/storyteller-platform/storyteller · Docs: https://storyteller-platform.dev/
- MIT, 138 yulduz, oxirgi push 2026-09-01, TypeScript, 6 ochiq issue. Docker bilan self-hosted.
- Kirish: EPUB + audiokitob (MP4/M4A/M4B/MP3), juftlab qo'yiladi → "Create readaloud". Chiqish: EPUB3 Media Overlays. **TTS dataset emas.**
- **Amaliy raqamlar (juda qimmatli)**:
  - Ko'pchilik kitob uchun **1–4 soat** ishlov, zamonaviy apparatda (https://storyteller-platform.dev/docs/managing/aligning).
  - Transkripsiya — eng resurstalab bosqich; **default holda mavjud barcha CPU yadro va butun RAM'ni oladi**. Docker'da `--memory=24g --cpus=8` kabi cheklov qo'yish tavsiya etiladi.
  - **"Turbo Mode" (v2.7.0+)**: boblarni kichik bo'laklarga bo'lib parallel ishlaydi → sezilarli tezroq, lekin **bo'lak chegaralarida aniqlik pasayadi**: timing siljishi, chegarada matn o'tkazib yuborilishi, qisqa boblarning butunlay tashlab ketilishi. Turbo'ni `1` ga qaytarib qayta hizalash bilan tuzatiladi.
  - GPU (CUDA yoki ROCm) transkripsiyani almashtira oladi; transkripsiyani boshqa mashinaga offload qilish ham mumkin.
- **Dars**: "uzun audioni parallellashtirish uchun bo'laklarga bo'lish" — bu aynan foydalanuvchi qiladigan ishtiyoqli optimizatsiya, va Storyteller'ning hujjatlari uning narxini ochiq yozib qo'ygan. easyaligner'ning "bir o'tishda" yondashuvi shu sababli afzal.

### 1.7 feldberlin/timething — `align-long` buyrug'i bor, lekin sokin
- URL: https://github.com/feldberlin/timething
- MIT, 133 yulduz, oxirgi push **2024-12-03** → ~1 yil 10 oy jimlik: **yarim tashlangan** deb belgilash kerak.
- Kirish strukturasi: papka ichida `text.csv` (audio yo'l + transkripsiya), `audio/` (MP3/WAV/FLAC/OGG), `aligned/` (chiqish).
- Chiqish: har audio ID uchun JSON — **char-level va word-level** hizalash + confidence + label.
- `align-long` buyrug'i aynan podkast/audiokitob uchun: ko'p soatli fayllarni batch'lab, parallellikni sozlash bilan.
- Wav2Vec2 asosidagi HF modellari: ingliz, nemis, golland, polyak, italyan, portugal, ispan, fransuz, rus, yapon, yunon, arab. **O'zbek yo'q** (MMS bilan almashtirish kerak bo'ladi).
- GPU va CPU ikkisida ishlaydi; batch size va worker thread sozlanadi.

### 1.8 readbeyond/aeneas — TASHLANGAN + AGPL
- URL: https://github.com/readbeyond/aeneas
- AGPL-3.0 (!), 2868 yulduz, 38 ochiq issue
- **TASHLANGAN**: oxirgi haqiqiy commitlar — 2020-05-13 (Debian paketi qo'shish), undan oldingisi 2017-03-16. Ya'ni **funksional rivojlanish 2017 da to'xtagan**. (`pushed_at` 2026-07-25 ko'rinadi, lekin bu commit tarixida yo'q — repo metadata harakati.)
- **AGPL-3.0** — agar foydalanuvchi tool'ni servis sifatida tarqatmoqchi bo'lsa, litsenziya jiddiy muammo.
- Ko'plab boshqa vositalar (afaligner, TTS-dataset-tools) hali ham unga bog'liq → ular ham aeneas'ning eskirganligini meros oladi.

### 1.9 Boshqa hizalash vositalari
| Vosita | URL | Holat |
|---|---|---|
| Montreal Forced Aligner (MFA) | https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner | MIT, 1897 ⭐, push 2026-08-20, **289 ochiq issue**. Windows: `conda create -c conda-forge -p C:\ProgramData\Miniconda3\envs\aligner montreal-forced-aligner=3.2.1`; rasmiy "Troubleshooting on Windows" bo'limi bor (ya'ni Windows og'riq bor deb tan olingan). Kaldi asosida; til uchun akustik model + talaffuz lug'ati kerak. |
| NVIDIA NeMo Forced Aligner (NFA) | https://github.com/NVIDIA-NeMo/Speech/tree/main/tools/nemo_forced_aligner · docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html | Apache-2.0. **1+ soatli audio uchun mo'ljallangan**; manifestda `duration` maydoni kerak emas va *"can align long audio files without running out of memory"*. `chunk_len_in_secs` (default 1.6) / `total_buffer_in_secs` (default 4.0) buffered streaming. Segmentlar `.` `?` `!` `...` bo'yicha; `additional_segment_grouping_separator` bilan sozlanadi. 14+ til. CTC modellar. |
| as-ideas/DeepForcedAligner | https://github.com/as-ideas/DeepForcedAligner | MIT, 81 ⭐, **2025-08-11 da ARXIVLANGAN**. Kirish allaqachon LJSpeech formatda bo'lishi kerak, chiqish char-duration `.npy` (FastSpeech trainingi uchun) → **segmentatsiya uchun emas**. Keys uchun mos emas. |
| lukerbs/forcealign | https://github.com/lukerbs/forcealign | Faqat **ingliz tili**. Uchinchi tomon non-Python bog'liqligi yo'q — o'rnatish oson. O'zbek uchun foydasiz. |
| kappa54m/audiobookutils | https://github.com/kappa54m/audiobookutils | 1 ⭐, push 2025-12-06, litsenziya NOASSERTION. Juda kichik shaxsiy loyiha — jiddiy asos emas. |

---

## 2. Whisper-asosli "avtomatik" TTS dataset generatorlari (transkript kerak emas)

Bu toifa keng tarqalgan, lekin **foydalanuvchining keysi uchun printsipial jihatdan noto'g'ri**: ular ground-truth matnni ishlatmaydi. Baribir ularning **segmentatsiya, filtratsiya va eksport qismlari** qayta ishlatishga arziydi.

### 2.1 devilismyfriend/ozen-toolkit — mashhur, lekin TASHLANGAN va litsenziyasiz
- URL: https://github.com/devilismyfriend/ozen-toolkit
- 86 ⭐, 24 fork, yaratilgan 2023-02-18, **oxirgi push 2023-08-19 → 3 yil jimlik: TASHLANGAN**
- **Litsenziya YO'Q** (`license: null`) → huquqiy jihatdan qayta ishlatish xavfli.
- `pyannote/segmentation` (HF token + litsenziya shartlarini qabul qilish kerak) + Whisper → LJ format: `wavs/` papkasi + `train.txt` / `valid.txt`.
- Windows-first: `Set Up Ozen.bat` va `Drag_Here.bat` — drag-and-drop workflow. UX jihatidan bu Windows foydalanuvchisi uchun **eng qulay pattern** va o'rganishga arziydi.
- 17 ochiq issue. Foydalanuvchilar hisobot qilgan muammolar (https://github.com/devilismyfriend/ozen-toolkit/issues):
  - "Installation failure: conflicts" — dependency to'qnashuvlari
  - "Where can I set the language for Whisper to analyse?" / "How can I fix a language..." — til parametri topilmaydi
  - "Which parameter sets the time between cuts?" — kesish parametrlari hujjatlashtirilmagan
  - "Control the length of AUDIOs generated????" — segment uzunligini boshqarish yo'q
  - "linux support", "eror", "always error ... help please" — umumiy nosozliklar
  - "Very detailed tutorial up-to-date" — hujjat yetishmasligi
- **UX darsi**: eng ko'p so'ralgan narsalar — **til tanlash, min/max segment uzunligi, kesish oralig'i**. Yangi tool'da bular birinchi ekranda ko'rinadigan parametr bo'lishi shart.

### 2.2 davidmartinrius/speech-dataset-generator — eng to'liq pipeline
- URL: https://github.com/davidmartinrius/speech-dataset-generator
- 263 ⭐, 203 commit, 7 ochiq issue. **Aralash litsenziya**: loyiha MIT, lekin whisperX BSD-4-Clause, yt-dlp litsenziyasiz, qolganlari MIT/Apache-2.0 → tarqatishda ehtiyot kerak.
- Bosqichlar: kirish (fayl / papka / YouTube / LibriVox / TED) → ixtiyoriy audio yaxshilash (**DeepFilterNet**, **Resemble-Enhance 0.0.1**, **Mayavoz**) → segmentatsiya + transkripsiya (**WhisperX 3.1.1**, **faster-whisper 1.0.0**) → **pyannote** embedding + diarization + gender aniqlash → **inaSpeechSegmenter 0.7.7** → **MOS** bahosi (**MOS < 3 bo'lgan segment rad etiladi**) → eksport.
- Chiqish formatlari: **LJSpeech** (default, `metadata.csv`), Metavoice, LibriSpeech (beta). Rejalashtirilgan: Common Voice 11, VoxPopuli, TED-LIUM, GigaSpeech, VCTK va h.k.
- Python 3.10+, GPU tavsiya (`--gpus all`).
- **KRITIK zaiflik**: *"tested and verified to work on Ubuntu 22"* — **macOS/Windows test qilinmagan**. Docker image hujjatda aytilgan bo'lsa ham hali mavjud emas.
- Boshqa cheklovlar (o'zi yozgan): `range_times` va sifat filtri tufayli audio 100% ishlatilmaydi; Resemble-Enhance "ba'zan bir oz buzuq" chiqaradi; Mayavoz 16kHz talab qiladi.
- **Nima qayta ishlatiladi**: MOS-asosida filtratsiya + denoise zanjiri + ko'p formatli eksporter g'oyasi. Kod emas (Linux-bound), arxitektura.

### 2.3 hclivess/speech-splitter — muammolarni eng aniq tushungan, lekin sinovdan o'tmagan
- URL: https://github.com/hclivess/speech-splitter
- MIT, **1 ⭐** (!), 100 commit, oxirgi push 2026-08-21, Windows/Linux/macOS prebuilt binary
- Aynan foydalanuvchi duch keladigan muammolarga qarshi mexanizmlar (README'dan):
  - **Musiqa/shovqin ifloslanishi**: YAMNet (AudioSet klassifikatori, onnxruntime'da) "music, singing, speech over music, noise, silence"ni filtrlaydi.
  - **Gap chegarasi**: barcha yozuv tizimlaridagi punktuatsiyani hurmat qilib bo'ladi, qisqa segmentlarni **birlashtiradi**, uzunlarni **eng uzun pauza joyidan** bo'ladi.
  - **So'z o'rtasida kesilish**: "cutoff detection" — birinchi/oxirgi so'zi kesilgan klipni aniqlab, skip yoki tag qiladi; **"boundary buffers that extend into the surrounding pause but never into the neighbouring word"** — bufer pauzaga kiradi, lekin qo'shni so'zga hech qachon kirmaydi. Bu juda to'g'ri dizayn qarori.
- faster-whisper (CTranslate2) word-level timestamps + Whisper confidence bahosi.
- Chiqish: WAV/FLAC/MP3 klip + har klip transkripti + `metadata.csv` (HF-mos) + JSON metadata + manifest + log.
- GPU: NVIDIA CUDA 12 (cuBLAS 12 + cuDNN 9), ilova ichidan yuklab olinadi (~1 GB). **10–30× realtime** o'rta darajali kartada, CPU'da **1–3×**.
- GUI tab'lar (model, segmentatsiya min/max + buferlar, sifat chegaralari, eksport, preset) + CLI ekvivalenti.
- **Zaiflik**: 1 yulduz — hech kim sinamagan. Lekin **README uning dizayn qarorlari uchun eng yaxshi o'quv materiali**.

### 2.4 Qolgan Whisper-generatorlari
| Vosita | URL | Metrikalar | Izoh |
|---|---|---|---|
| thorstenMueller/Audio-to-Voice-Dataset | https://github.com/thorstenMueller/Audio-to-Voice-Dataset | MIT, 39 ⭐, push **2024-09-28** (2 yil → sokin) | `create_ljspeech.py`: Whisper transkripsiya + sukunat bo'yicha gap-gap bo'lish → LJSpeech. Sodda referens. Muallif Thorsten Müller — nemis TTS dataset jamoasida tanilgan. |
| gokhaneraslan/tts-dataset-generator | https://github.com/gokhaneraslan/tts-dataset-generator | Apache-2.0, 17 ⭐, 6 fork | Video/audio → sukunat asosida segment → Whisper → `MyTTSDataset/wavs/` + `metadata.csv`. **Windows qo'llanmasi bor** (choco/scoop bilan ffmpeg). Diarization/speaker filtri **yo'q** → faqat bitta so'zlovchi. Oxirgi commit: ma'lumot topilmadi. |
| JarodMica/audiosplitter_whisper | https://github.com/JarodMica/audiosplitter_whisper | Yulduz/commit: ma'lumot topilmadi | WhisperX asosida, <10s klip, RVC uchun mo'ljallangan. Python 3.10, ffmpeg, CUDA tavsiya (CPU'da "juda sekin"). Windows ko'rsatmalari mavjud (execution policy va h.k.). **Litsenziya README'da ko'rsatilmagan**. AMD/Mac to'liq emas. |
| hollygrimm/voice-dataset-creation | https://github.com/hollygrimm/voice-dataset-creation | 73 ⭐, push 2026-05-06, **litsenziya ko'rsatilmagan** | Kam resursli tillar uchun (Māori, Plains Cree). 5 bosqich: yozib olish/segmentatsiya (Audacity sukunat yoki **qo'lda belgilash**) → SNR sifat tahlili → transkripsiya (Whisper / **MMS** / **qo'lda**) → augmentatsiya → LJSpeech eksport. **Aniq inson ko'rigi bosqichi bor**: "Review transcripts" + `knowledge_keeper_reviewed` metadata maydoni. Windows: `.venv\Scripts\activate` ko'rsatilgan; Python 3.11+, uv, ffmpeg, sox. |
| sushant-t/tts-trainer | https://github.com/sushant-t/tts-trainer | Metrikalar: ma'lumot topilmadi | Sukunat detektsiyasi + Whisper → LJSpeech. |

---

## 3. Brauzerdagi review/tahrir vositalari (UX uchun eng yaqin namunalar)

### 3.1 FS-17/SpeechDataBuilder
- URL: https://github.com/FS-17/SpeechDataBuilder · Demo: https://fs-17.github.io/SpeechDataBuilder/
- MIT, 11 ⭐, 6 commit, v2.0 (React 19 + TypeScript)
- **To'liq brauzerda, server-side ishlov YO'Q**; transkript va sozlamalar `localStorage`da.
- Transkripsiya esa **tashqi** — Google AI Studio (Gemini) yoki OpenAI (Whisper) API. Ya'ni **audio cloudga yuboriladi** → maxfiylik/narx muammosi, lokal model yo'q.
- Waveform vizualizatsiya + **region selection** (transkript hizalash uchun), trim-to-selection, Ctrl+S auto-save.
- Eksport: LJSpeech, Common Voice-uslubidagi CSV, custom CSV/JSON/TXT (ajratgich sozlanadi), ixtiyoriy ZIP.
- Fayl hajmi / audio davomiyligi cheklovi hujjatlashtirilmagan → 10 soatli fayl brauzerda muammo bo'lishi ehtimoli yuqori.

### 3.2 williampepple1/voice-studio
- URL: https://github.com/williampepple1/voice-studio
- **0 ⭐, 3 commit, litsenziya ko'rsatilmagan** → hali prototip, ishonch bo'lmasin. Lekin **funksiya ro'yxati aynan kerakli review UX**:
  - drag-and-drop audio import + avtomatik nutq segmentini aniqlash
  - waveform tahrir: qo'lda kesish, **zero-crossing'ga mos bo'lish nuqtalari**, trim
  - har klipni **loop playback** bilan eshitib transkriptni kiritish/tekshirish
  - `.txt` / `.csv` dan skript import (ya'ni ground-truth matnni olib kirish)
  - brauzer nutqni tanish bilan diktovka
  - **de-reverb / xona tozalash** sozlanadigan DSP presetlari
  - eksport: ZIP ichida juftlangan WAV/TXT + ixtiyoriy `metadata.csv` (LJSpeech) + `manifest.json`; sample rate, mono/stereo, peak normalizatsiya, **micro-fade**lar
- Web Audio API — asosiy ishlov lokal brauzerda. React 19 / TypeScript / Vite / Tailwind, Node.js 22+.
- Audio uzunligi cheklovi hujjatlashtirilmagan.
- **Xulosa**: kod emas, **UX spetsifikatsiyasi** sifatida ko'chirib olish kerak. "Micro-fade" va "zero-crossing split" — tajribali detallar (klik/pop shovqinini oldini oladi).

---

## 4. Segmentatsiya va audio yordamchi vositalar

| Vosita | URL | Ma'lumot |
|---|---|---|
| openvpi/audio-slicer | https://github.com/openvpi/audio-slicer | MIT, 885 ⭐, oxirgi push **2024-06-08** (2 yil → sokin, lekin kod kichik va barqaror). **RMS-asosida** sukunat detektsiyasi: chegara ostidagi barcha frame sukunat; minimal davomdan uzun sukunat oraliqlari topilib, **eng jim nuqtadan** bo'linadi. **Transkripsiya qilmaydi.** CLI + Python API; GUI alohida repo. i7-8750H CPU'da **400×+ realtime**. Zaiflik: shovqinli audioda parametrlarni qo'lda sozlash kerak; audio formatlari hujjatlashtirilmagan. |
| m-bain/whisperX | https://github.com/m-bain/whisperx | BSD-2-Clause, **24 294 ⭐**, push 2026-09-26, **228 ochiq issue**. wav2vec2 hizalash bilan so'z darajasidagi timestamp; large-v2 bilan 70× realtime, `beam_size=5` da **<8GB VRAM**. Default modellar: en, fr, de, es, it (torchaudio pipeline). **README'da e'lon qilingan cheklovlar**: lug'atda bo'lmagan belgili so'zlar (`"2014."`, `"£13.60"`) hizalanmaydi; overlapping speech yaxshi ishlanmaydi; diarization mukammal emas; har til uchun alohida wav2vec2 modeli kerak. |
| linto-ai/whisper-timestamped | https://github.com/linto-ai/whisper-timestamped | Qo'shimcha model talab qilmaydi — Whisper'ning timestamp token ehtimolliklaridan foydalanadi. Yulduz/commit: ma'lumot topilmadi. Ochiq issue #173: fine-tune qilingan Whisper modellarida timestamp prognozi buziladi — **muhim ogohlantirish** agar o'zbek uchun fine-tune Whisper ishlatilsa. |
| Purfview/whisper-standalone-win | https://github.com/Purfview/whisper-standalone-win | **Windows uchun eng qulay yechim**: Faster-Whisper-XXL standalone `.exe` — barcha kutubxonalar ichida, **Python kerak emas**. x86-64, Windows 10+. CUDA aniqlansa avtomatik GPU'ga o'tadi. CLI (Subtitle Edit, Tero Subtitler, FFAStrans kabi programmalardan chaqiriladi). "Pro" versiya — donorlar uchun yopiq. |

---

## 5. Sanoat darajasidagi pipeline va korpus kutubxonalari

### 5.1 NVIDIA/NeMo-speech-data-processor (SDP) — arxitektura namunasi sifatida eng qimmatli
- URL: https://github.com/NVIDIA/NeMo-speech-data-processor · Docs: https://nvidia.github.io/NeMo-speech-data-processor/
- Apache-2.0, 221 ⭐, 48 fork, Python 3.10 rasmiy qo'llanadi
- **Falsafa**: har operatsiya — bitta "processor" klassi. Har processor **NeMo-uslubidagi manifest**ni (JSON lines: `audio_filepath`, `text`, `duration`) o'qiydi, o'zgartiradi va **yangi manifest faylini yozadi**. Qaysi processorlar ishga tushishi **YAML config**da e'lon qilinadi.
- Mavjud processorlar (misol): `CreateInitialManifestMCV`, **`ASRInference`** (ASR bilan qayta transkripsiya qilib, **reference matndan juda ko'p farq qilgan utterance'larni tashlab yuboradi** — bu foydalanuvchi keysida hizalash sifatini avtomatik tekshirishning aniq usuli), `SubRegex` (regex matn almashtirish), `DropNonAlphabet`, `KeepOnlySpecifiedFields`.
- GPU faqat ASR inference uchun kerak; qolgani CPU.
- OS cheklovi hujjatlashtirilmagan (Python + YAML → Windows'da ishlashi kutiladi).
- **Granary** pipeline (25 Yevropa tili uchun pseudo-labelling) shu repo ichida: https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/dataset_configs/multilingual/granary
- **Eng muhim dars**: "har bosqich → yangi manifest fayli" pattern'i **tekin checkpoint/resume** beradi. Foydalanuvchi shu pattern'ni to'g'ridan-to'g'ri ko'chirishi kerak.

### 5.2 lhotse-speech/lhotse
- URL: https://github.com/lhotse-speech/lhotse · Docs: https://lhotse.readthedocs.io/
- Apache-2.0, **1.2k ⭐**, Python 3.10+
- Asosiy abstraksiyalar: **Recording** (sampling rate, kanal, davomiylik), **SupervisionSet / SupervisionSegment** (transkript, so'zlovchi, til), **CutSet** (asosiy ish quroli — moslashuvchan manipulyatsiya va on-the-fly transformatsiya).
- Uzun korpuslar uchun: mixing/truncation/padding **on-the-fly** bajariladi → saqlash joyi minimallashtiriladi; **lazy manifest**lar (`jsonl.gz`); **Lhotse Shar** — ketma-ket I/O va modulyarlik uchun optimallashtirilgan saqlash formati; blending, bucketing, ko'p-node training uchun randomizatsiya.
- Kaldi data dir import/export qo'llanadi.
- **Forced alignment o'zi qilmaydi** (hizalash natijalarini saqlaydi).
- **Foydalanuvchi uchun qiymati**: 10 soat bitta ma'ruzachi uchun Lhotse **ortiqcha** (over-engineering). Lekin `SupervisionSegment` sxemasi (recording_id, start, duration, channel, text, speaker, language, alignment) — **oraliq ma'lumot modeli uchun tayyor, sinovdan o'tgan sxema**. Uni ko'chirib olish arziydi.

### 5.3 icefall / espnet — "stage" pattern
- icefall: https://github.com/k2-fsa/icefall · Data prep: https://k2-fsa.github.io/icefall/for-dummies/data-preparation.html — har recipe'da `prepare.sh`, `--stage N --stop-stage M` bilan bosqichdan davom etish. Lhotse manifestlarini ishlatadi.
- espnet: https://github.com/espnet/espnet · TTS recipe: https://espnet.github.io/espnet/recipe/tts1.html — Kaldi uslubidagi `run.sh --stage`.
- **Dars**: akademik nutq pipeline'larining 20 yillik konvensiyasi — **raqamlangan, qayta ishga tushirilishi mumkin bosqichlar**. Bu monolit skriptdan ko'ra ancha ishonchli.

### 5.4 Coqui TTS dataset vositalari
- **coqui-ai/TTS** — https://github.com/coqui-ai/TTS — MPL-2.0, 46 077 ⭐, **oxirgi push 2024-08-16**. Kompaniya yopilgan; repo arxivlanmagan lekin **amalda tashlangan**.
- **idiap/coqui-ai-TTS** — https://github.com/idiap/coqui-ai-TTS — MPL-2.0, 2332 ⭐, **oxirgi push 2026-06-10 → TIRIK FORK**. PyPI: `coqui-tts`. Docs: https://coqui-tts.readthedocs.io/ — **yangi loyihalar shu forkni ishlatishi kerak**.
- `TTS/bin/` ichidagi dataset skriptlari (tasdiqlangan): `find_unique_chars.py`, `find_unique_phonemes.py`, `compute_statistics.py`, `remove_silence_using_vad.py`, `extract_tts_spectrograms.py`, `resample.py`.
- `notebooks/dataset_analysis/` (https://github.com/coqui-ai/TTS/tree/dev/notebooks/dataset_analysis):
  - **AnalyzeDataset** — klip va transkript uzunliklari taqsimoti; outlier'larni topish (juda uzun, qisqa matn + uzun audio va h.k.) va trainingdan oldin olib tashlash.
  - **CheckSpectrograms** — klip shovqin darajasini o'lchash va yaxshi audio ishlov parametrlarini topish.
- Formatterlar: `TTS/tts/datasets/formatters.py` — `ljspeech`, `mailabs`, `vctk` va boshqa o'nlab format o'qish funksiyalari. **Yangi tool'ning chiqishi shu formatterlardan biri (`ljspeech`) bilan hech sozlamasdan o'qilishi kerak** — bu qattiq talab.

### 5.5 huggingface/dataspeech — sifat atributlari annotatsiyasi
- URL: https://github.com/huggingface/dataspeech
- MIT, 401 ⭐, **oxirgi push 2024-09-03 → 2 yil jimlik: SOKIN/yarim tashlangan**, 20 ochiq issue
- **Dataset yaratmaydi — mavjudini annotatsiya va filtrlaydi.** Kirish: HF `datasets` (audio kolonka + matn kolonka). Chiqish: yangi metadata kolonkalari bilan HF Hub'ga push.
- Hisoblaydigan uzluksiz o'zgaruvchilar: **speaking rate** (fonema / utterance uzunligi), **SI-SDR**, **C50** (reverberatsiya), **pitch std/mean** (nutq monotonligi), **SNR**, **PESQ**.
- Ulardan tabiiy tildagi ta'rif yasaydi ("slightly slowly, very monotone") — Parler-TTS fine-tune uchun. Bitta va ko'p so'zlovchi ikkisini qo'llaydi.
- GPU tavsiya (operatsiyalar avtomatik masshtablanadi), CPU'da ham ishlaydi lekin sekin.
- **Foydalanuvchi uchun**: 10 soatlik bitta ovoz uchun SNR/C50/PESQ hisoblab **segmentlarni triage qilish** (qaysi birini birinchi ko'rish kerak) g'oyasi juda mos. Lekin kod sokin → metrika ro'yxatini olib, o'z hisobini yozish tezroq.

---

## 6. Yozib olish workflow (record-a-prompt)

### rhasspy/piper-recording-studio
- URL: https://github.com/rhasspy/piper-recording-studio
- MIT, 223 ⭐, oxirgi push **2026-08-10 → TIRIK**
- Lokal web ilova: foydalanuvchi brauzerda tilni tanlaydi, ko'rsatilgan prompt'ni o'qib yozib oladi, yozuv lokal saqlanadi.
- Prompt formati: `<id>\t<text>` — til papkalari ichida. Chiqish: `output/<language>/`.
- **Eksport**: `python3 -m export_dataset output/<language>/ /path/to/dataset` → **LJSpeech** (Piper-mos). Eksportda tugma bosilishi / klaviatura shovqinini olib tashlash uchun sukunat detektsiyasi parametrlari bor.
- Docker yoki standalone Python. **Windows aniq qo'llanishi hujjatlashtirilmagan** (ko'rsatmalar Linux/Unix uchun).
- Piper training hujjati: https://github.com/rhasspy/piper/blob/master/TRAINING.md — klip **3–15 sekund**; `--batch-size 32 --max-phoneme-ids 400` ≈ 24 GB VRAM; `--max-phoneme-ids` N dan katta gaplarni tashlab yuboradi; **mavjud modeldan fine-tune tavsiya** (bir xil audio sifat va sample rate kerak, bir xil til shart emas).
- **Muhim UX kontrasti**: foydalanuvchi keysi "record-a-prompt" EMAS (audio allaqachon yozilgan). Lekin piper-recording-studio **lokal web UI + CLI eksporter** ajratmasi — aynan tavsiya etiladigan arxitektura.

---

## 7. RVC / so-vits-svc ekotizimi dataset vositalari

**Muhim ogohlantirish**: RVC va so-vits-svc — voice *conversion*, ular **transkriptni umuman ishlatmaydi**. Shuning uchun ularning preprocessing'i TTS dataset uchun **yarim yaroqli**: audio tozalash/kesish/resample qismi foydali, matn qismi yo'q.

| Vosita | URL | Holat | Dataset qismi |
|---|---|---|---|
| IAHispano/Applio | https://github.com/IAHispano/Applio · docs: https://docs.applio.org/getting-started/training/ | MIT, **3767 ⭐**, push **2026-09-29** (juda tirik) | GUI'da "Pre-process Dataset" tugmasi. Shovqin bo'lsa **Vocals/Accompaniment Separation + dereverb** ('HP2-all-vocals' UVR modeli) bilan tozalash tavsiya etiladi. Min ~10 daqiqa audio tavsiya. **Windows'da keng ishlatiladi** (AI Hub hamjamiyati: https://docs.aihub.gg/rvc/local/applio/). |
| RVC-Project/Retrieval-based-Voice-Conversion-WebUI | https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI | MIT, **38 614 ⭐**, push 2026-08-04 | Asl RVC WebUI. Gradio. |
| Mangio621/Mangio-RVC-Fork | https://github.com/Mangio621/Mangio-RVC-Fork | MIT, 1233 ⭐, oxirgi push **2023-09-27 → TASHLANGAN (3 yil)** | CREPE + hybrid f0. Applio uning vorisi. **Ishlatilmasin.** |
| svc-develop-team/so-vits-svc | https://github.com/svc-develop-team/so-vits-svc | — | Dataset oqimi: `dataset_raw/<speaker>/*.wav` → `python resample.py --skip_loudnorm` (44100 Hz) → `python preprocess_flist_config.py --speech_encoder vec768l12` (train/val/test split + config avtomatik generatsiya, `n_speakers` avtomatik) → `python preprocess_hubert_f0.py` (hubert + f0). 7 speech encoder tanlovi. |
| voicepaw/so-vits-svc-fork | https://github.com/voicepaw/so-vits-svc-fork | Realtime + yaxshilangan interfeys fork | — |

**Qayta ishlatishga arzigulik g'oya**: so-vits-svc'ning `preprocess_flist_config.py` pattern'i — **train/val/test bo'linishi va config generatsiyasi bitta determinized skript** bo'lishi. Bu qo'lda bo'linish xatolarini yo'q qiladi.

---

## 8. Umumiy taqqoslash jadvali

| Vosita | Nima qiladi | Transkript kerakmi? | Windows OK? | Qo'llab-quvvatlanadimi? | Havola |
|---|---|---|---|---|---|
| **kb-labb/easyaligner** | Soatlab audioni to'liq matnga bir o'tishda hizalash, word-level JSON + confidence | **HA (ground truth)** | Ehtimol ha (pip + PyTorch); aniq test: ma'lumot topilmadi | ✅ Tirik (2026-07-04), MIT, 24 ⭐ | https://github.com/kb-labb/easyaligner |
| **jianfch/stable-ts** | `align()` / `align_words()` / `refine()` / `regroup()` — matn↔audio hizalash va segment qayta guruhlash | HA (`align` uchun) | ✅ (sof Python/PyTorch) | ⚠️ **ARXIVLANGAN 2026-05-30**, MIT, 2281 ⭐ | https://github.com/jianfch/stable-ts |
| **NVIDIA NFA** | CTC ASR bilan token/word/segment timestamp, 1+ soatli audio OOM'siz | HA yoki ASR generatsiyasi | Ehtimol (NeMo o'rnatish og'ir) | ✅ Tirik, Apache-2.0 | https://github.com/NVIDIA-NeMo/Speech/tree/main/tools/nemo_forced_aligner |
| **NeMo SDP** | YAML-konfiguratsiyali processor zanjiri, manifest→manifest; `ASRInference` bilan noto'g'ri hizalanganni tashlash | HA | Ehtimol (Python 3.10) | ✅ Tirik, Apache-2.0, 221 ⭐ | https://github.com/NVIDIA/NeMo-speech-data-processor |
| **youmebangbang/TTS-dataset-tools** | Matn + uzun audio → aeneas force-align → kesimlar + proofread UI | **HA** | ⚠️ Ha, lekin aeneas'da xotira muammosi — uzun kesim qilolmaydi | ❓ Oxirgi commit: ma'lumot topilmadi, MIT, 52 ⭐ | https://github.com/youmebangbang/TTS-dataset-tools |
| **r4victor/afaligner** | DTW (sintez+DTW) bilan matn↔audio sync map (JSON/SMIL) | HA | ❌ **README: Windows qo'llanmaydi** | ✅ Tirik (2025-08-09), MIT, 105 ⭐ | https://github.com/r4victor/afaligner |
| **r4victor/syncabook** | Matn + audio → EPUB3 Media Overlays (TTS dataset EMAS) | HA | ❌ (afaligner orqali) / Docker | ✅ MIT, 360 ⭐ | https://github.com/r4victor/syncabook |
| **smoores-dev/storyteller** | EPUB + audiokitob → hizalangan EPUB3 (TTS dataset EMAS) | HA (EPUB) | ✅ Docker | ✅ Tirik (2026-09-01), MIT, 138 ⭐ | https://github.com/smoores-dev/storyteller |
| **feldberlin/timething** | `align-long` — ko'p soatli audio↔matn, char+word JSON | HA (`text.csv`) | Ehtimol | ⚠️ Sokin (2024-12-03), MIT, 133 ⭐ | https://github.com/feldberlin/timething |
| **readbeyond/aeneas** | Klassik forced aligner (Python/C) | HA | ⚠️ Ha, lekin uzun kesimda xotira muammosi | ❌ **TASHLANGAN** (funksional commit 2017/2020), **AGPL-3.0**, 2868 ⭐ | https://github.com/readbeyond/aeneas |
| **MFA** | Kaldi asosidagi forced aligner, TextGrid | HA + lug'at/akustik model | ⚠️ conda-forge orqali; rasmiy Windows troubleshooting bor | ✅ Tirik (2026-08-20), MIT, 1897 ⭐, 289 issue | https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner |
| **as-ideas/DeepForcedAligner** | LJSpeech→char duration `.npy` (training uchun) | HA | Ehtimol | ❌ **ARXIVLANGAN 2025-08-11**, MIT, 81 ⭐ | https://github.com/as-ideas/DeepForcedAligner |
| **devilismyfriend/ozen-toolkit** | Audio → pyannote+Whisper → LJ format | YO'Q | ✅ `.bat` drag-drop | ❌ **TASHLANGAN 2023-08-19**, **litsenziya yo'q**, 86 ⭐ | https://github.com/devilismyfriend/ozen-toolkit |
| **davidmartinrius/speech-dataset-generator** | Eng to'liq pipeline: denoise+WhisperX+pyannote+MOS→LJSpeech | YO'Q | ❌ Faqat Ubuntu 22 test qilingan | ✅ Tirik, MIT+aralash, 263 ⭐ | https://github.com/davidmartinrius/speech-dataset-generator |
| **hclivess/speech-splitter** | faster-whisper + YAMNet musiqa filtri + cutoff detection → metadata.csv | YO'Q | ✅ Prebuilt binary | ✅ Tirik (2026-08-21), MIT, **1 ⭐ (sinovdan o'tmagan)** | https://github.com/hclivess/speech-splitter |
| **gokhaneraslan/tts-dataset-generator** | Audio/video → sukunat segment → Whisper → LJSpeech | YO'Q | ✅ (choco/scoop ffmpeg) | ❓ Apache-2.0, 17 ⭐ | https://github.com/gokhaneraslan/tts-dataset-generator |
| **thorstenMueller/Audio-to-Voice-Dataset** | `create_ljspeech.py`: Whisper + sukunat → LJSpeech | YO'Q | Ehtimol | ⚠️ Sokin (2024-09-28), MIT, 39 ⭐ | https://github.com/thorstenMueller/Audio-to-Voice-Dataset |
| **hollygrimm/voice-dataset-creation** | 5 bosqich + **aniq inson ko'rigi** + LJSpeech | Ixtiyoriy (Whisper/MMS/qo'lda) | ✅ `.venv\Scripts\activate` | ✅ Tirik (2026-05-06), litsenziya yo'q, 73 ⭐ | https://github.com/hollygrimm/voice-dataset-creation |
| **openvpi/audio-slicer** | RMS sukunat asosida kesish (transkripsiya yo'q), 400× realtime | YO'Q | ✅ (numpy/librosa/soundfile) | ⚠️ Sokin (2024-06-08), MIT, 885 ⭐ | https://github.com/openvpi/audio-slicer |
| **m-bain/whisperX** | Whisper + wav2vec2 word timestamps + diarization | YO'Q | ✅ | ✅ Tirik (2026-09-26), BSD-2, 24.3k ⭐ | https://github.com/m-bain/whisperx |
| **Purfview/whisper-standalone-win** | Faster-Whisper-XXL standalone `.exe`, Python kerak emas | YO'Q | ✅ **Windows-native** | ✅ Tirik | https://github.com/Purfview/whisper-standalone-win |
| **rhasspy/piper-recording-studio** | Record-a-prompt web UI → `export_dataset` → LJSpeech | Prompt matn kerak | ⚠️ Docker/Python (Linux ko'rsatmalari) | ✅ Tirik (2026-08-10), MIT, 223 ⭐ | https://github.com/rhasspy/piper-recording-studio |
| **idiap/coqui-ai-TTS** | `find_unique_chars`, `compute_statistics`, `remove_silence_using_vad`, `resample`, AnalyzeDataset/CheckSpectrograms | HA (LJSpeech) | ✅ | ✅ **Tirik fork** (2026-06-10), MPL-2.0, 2332 ⭐ | https://github.com/idiap/coqui-ai-TTS |
| **coqui-ai/TTS** (asl) | yuqoridagi bilan bir xil | HA | ✅ | ❌ **Amalda tashlangan** (2024-08-16), MPL-2.0, 46k ⭐ | https://github.com/coqui-ai/TTS |
| **lhotse-speech/lhotse** | CutSet/Supervision korpus manipulyatsiyasi, lazy manifest, Shar | HA (supervision) | Ehtimol | ✅ Tirik, Apache-2.0, 1.2k ⭐ | https://github.com/lhotse-speech/lhotse |
| **huggingface/dataspeech** | SNR/C50/PESQ/speaking-rate/pitch annotatsiyasi (yaratish emas) | HA | Ehtimol | ⚠️ Sokin (2024-09-03), MIT, 401 ⭐ | https://github.com/huggingface/dataspeech |
| **IAHispano/Applio** | RVC dataset preprocess + UVR vocal separation + dereverb | YO'Q (RVC) | ✅ | ✅ Juda tirik (2026-09-29), MIT, 3767 ⭐ | https://github.com/IAHispano/Applio |
| **Mangio621/Mangio-RVC-Fork** | RVC fork, CREPE f0 | YO'Q | ✅ | ❌ **TASHLANGAN 2023-09-27**, 1233 ⭐ | https://github.com/Mangio621/Mangio-RVC-Fork |
| **FS-17/SpeechDataBuilder** | Brauzerda dataset quruvchi, waveform region, LJSpeech eksport | Ixtiyoriy (tashqi AI API) | ✅ brauzer | ⚠️ 11 ⭐, 6 commit, MIT | https://github.com/FS-17/SpeechDataBuilder |
| **williampepple1/voice-studio** | Brauzerda segment tahriri + loop proofread + de-reverb + LJSpeech ZIP | Ixtiyoriy (script import) | ✅ brauzer | ⚠️ **0 ⭐, 3 commit, prototip**, litsenziya yo'q | https://github.com/williampepple1/voice-studio |
| **Label Studio** | Umumiy annotatsiya: ASR-with-segments, audio transcription, diarization shablonlari | — | ✅ pip (Gohlke wheel eslatmasi) | ✅ Tirik, Apache-2.0, 28.4k ⭐ | https://github.com/HumanSignal/label-studio |
| **midas-research/audino** | Ochiq audio annotatsiya: VAD, diarization, ASR, emotion | — | Docker/web | ✅ Tirik (2026-02-03), **litsenziya NOASSERTION**, 1148 ⭐, 58 issue | https://github.com/midas-research/audino |

### TASHLANGAN / ARXIVLANGAN deb belgilanganlar (aniq ro'yxat)
| Loyiha | Oxirgi haqiqiy harakat | Izoh |
|---|---|---|
| **readbeyond/aeneas** | 2020-05-13 (funksional: 2017-03) | + AGPL-3.0. Lekin hali ham boshqa vositalarning poydevorida turadi. |
| **devilismyfriend/ozen-toolkit** | 2023-08-19 | + litsenziya yo'q + 17 hal qilinmagan issue |
| **Mangio621/Mangio-RVC-Fork** | 2023-09-27 | Applio uning vorisi |
| **coqui-ai/TTS** | 2024-08-16 | Kompaniya yopilgan → `idiap/coqui-ai-TTS` ishlatilsin |
| **as-ideas/DeepForcedAligner** | 2025-08-11 da **arxivlangan** | read-only |
| **jianfch/stable-ts** | 2026-05-30 da **arxivlangan** | read-only, lekin kod hali ishlatilishga yaroqli |
| **huggingface/dataspeech** | 2024-09-03 | 2 yil jimlik, 20 ochiq issue |
| **openvpi/audio-slicer** | 2024-06-08 | Kod kichik va tugallangan — "sokin" deb qarash mumkin |
| **feldberlin/timething** | 2024-12-03 | ~1 yil 10 oy |
| **thorstenMueller/Audio-to-Voice-Dataset** | 2024-09-28 | Demo/referens sifatida |

---

## 9. Annotatsiya va human-in-the-loop vositalari

### 9.1 Label Studio (https://github.com/HumanSignal/label-studio)
- Apache-2.0, 28.4k ⭐. Windows'da `pip install` ishlaydi (README'da Gohlke wheel'lari haqida maxsus eslatma bor).
- Tegishli tayyor shablonlar:
  - **Automatic Speech Recognition using Segments** — https://labelstud.io/templates/automatic_speech_recognition_segments — audioni segmentlab, har segmentni transkripsiya qilish. **Aynan kerakli review ekrani.**
  - Audio Transcription — https://labelstud.io/templates/transcribe_audio
  - Speaker Diarization — https://labelstud.io/templates/speaker_segmentation
  - Audio Classification with Segments — https://labelstud.io/templates/audio_regions
  - Butun galereya — https://labelstud.io/templates/gallery_asr
- Tag konfiguratsiyasi bilan tahrirlanadigan transkript, majburiy maydonlar, segmentga xos annotatsiyalar sozlanadi.
- **Muhim cheklov**: HumanSignal o'z hujjatida Enterprise versiyada *"advanced audio transcription interface built to support faster, more precise annotation at scale"* borligini aytadi (https://docs.humansignal.com/templates/transcribe_audio) → **OSS versiyadagi audio UI ko'p hajmli, tez review uchun sekinroq**.
- Zaiflik: umumiy maqsadli platforma → 10 soat audio uchun o'rnatish/konfiguratsiya narxi katta, klaviatura bilan tez o'tish (keyboard-first triage) oqimi TTS uchun optimallashtirilmagan.

### 9.2 midas-research/audino (https://github.com/midas-research/audino)
- 1148 ⭐, push 2026-02-03, TypeScript, **litsenziya NOASSERTION** (ishlatishdan oldin tekshirish kerak), 58 ochiq issue. Human Protocol homiyligida.
- VAD, diarization, speaker ID, ASR, emotion recognition uchun transkripsiya + label. Ko'p tilli, moslashuvchan label yaratish, maxsus format eksporti.
- **v2.0 hali development'da**, kelajakda v1 dan v2 ga to'liq ko'chish rejalashtirilgan → hozir ikki tizim orasida.
- Ilmiy maqola: https://arxiv.org/pdf/2006.05236

### 9.3 doccano (https://github.com/doccano/doccano)
- **Bu matn annotatsiya vositasi** (NER, klassifikatsiya, seq2seq) — **audio segmentatsiya/review uchun MOS EMAS**. Faqat matnni oldindan tozalash/normallashtirish bosqichida foydali bo'lishi mumkin. Keys uchun ortiqcha.

### 9.4 ELAN (https://archive.mpi.nl/tla/elan · yuklab olish: https://archive.mpi.nl/tla/elan/download)
- Max Planck Institute for Psycholinguistics (Nijmegen). Bepul, Windows/macOS/Linux, Java.
- Cheksiz sonli matn annotatsiyasi, ko'p qatlamli ("tier") struktura. Audio+video.
- Zaiflik: lingvistik tadqiqot uchun mo'ljallangan — **minglab TTS segmentini tez tasdiqlash uchun sekin**; ishlatilganda iqtibos keltirish talab qiladi.

### 9.5 Praat (https://www.fon.hum.uva.nl/praat/)
- Paul Boersma / David Weenink, Amsterdam universiteti. Bepul, barcha OS.
- **TextGrid** — interval tier (chegaralari bilan ketma-ket intervallar) va point tier: https://www.fon.hum.uva.nl/praat/manual/TextGrid.html
- **`LongSound` obyekti** — muhim: uzun audio faylni **to'liq RAM'ga yuklamasdan** diskdan o'qib ko'rish va annotatsiya qilish. `LongSound: To TextGrid...` — https://www.fon.hum.uva.nl/praat/manual/LongSound__To_TextGrid___.html
- `Sound: To TextGrid (speech activity)...` — avtomatik VAD segmentatsiyasi: https://www.fon.hum.uva.nl/praat/manual/Sound__To_TextGrid__speech_activity____.html
- MFA'ning standart chiqishi TextGrid → Praat bilan bevosita ko'riladi.
- Zaiflik: UI eskirgan, skriptlash tili o'ziga xos; ommaviy review uchun emas.

### 9.6 Audacity labels — jamoada eng ko'p ishlatilgan amaliy usul
Real workflow (rhasspy/piper discussion #191 dan, quyida batafsil):
1. Bir nechta MP3 ni Audacity'da birlashtirish
2. **Analyze → "Label Sounds"** — kamida **0.6 s** sukunat, minimal **3 s** segment
3. **22 050 Hz WAV** eksport (File → Export → Export Multiple, har label alohida fayl) yoki File → Export → **Export Labels** (TSV: start, end, label)
4. Sox bilan **>15 s** yozuvlarni olib tashlash
5. Whisper bilan transkripsiya → `metadata.csv`

Bu bepul, Windows'da ishlaydi va **hech qanday o'rnatish og'rig'i yo'q** — shu sababli hamjamiyatda eng keng tarqalgan.

### 9.7 Inson ko'rigi amalda qancha vaqt oladi?
Nashr etilgan vaqt smetalari:
- **Qo'lda to'liq transkripsiya**: 1 soat audio uchun **4–6 soat**; maxsus dasturlar (foot pedal, text expander) bilan professional **2–3 soat**; yomon audio (shovqin, bir-biriga o'tuvchi so'zlovchilar, kuchli aksent) **6–8 soat** — https://www.gmrtranscription.com/blog/transcribing-an-hour-of-audio
- **ASR + inson tuzatishi (post-editing)**: avtomatik transkripsiya bir necha daqiqada tayyor, keyin **15–20 daqiqa** ko'rik kerak; **toza audio** uchun **5–10 daqiqa** / yozilgan soat — https://verbit.ai/how-long-it-really-takes-to-transcribe-accurate-audio/ · https://www.sky-scribe.com/en/blog/ai-audio-transcription-accuracy-benchmarks-fixes
- Murakkab kontent uchun professional aniqlikka yetish odatda audio uzunligining **2–4×** inson tahririni talab qiladi.
- LXT kabi provayderlar buni alohida xizmat sifatida sotadi: https://www.lxt.ai/services/post-editing-of-asr/

**Foydalanuvchi keysi uchun muhim nuans**: unda **matn allaqachon bor**. Shu sababli review vazifasi "so'zlar to'g'rimi?" dan "**kesim to'g'rimi?**" ga o'zgaradi — bu ancha tez ish. Confidence ball bo'yicha triage qilinsa (past ballli 10–15% segmentni ko'rish), 10 soat audio uchun **~1–3 soat** ko'rik realistik smeta. Barcha segmentni qo'lda eshitib chiqish esa 10 soat + bo'ladi va buni qilish shart emas.

---

## 10. Tijorat / boshqariladigan variantlar (qisqa, taqqoslash uchun)

| Xizmat | Nima talab qiladi / beradi | Narx |
|---|---|---|
| **ElevenLabs Professional Voice Clone** — https://elevenlabs.io/docs/eleven-creative/voices/voice-cloning/professional-voice-cloning | **Minimum 30 daqiqa**, optimal natija uchun **2–3 soat** audio. Texnik: WAV **44.1 yoki 48 kHz, ≥24 bit**. Faqat bitta ovoz; bitta uslub. Ogohlantirish: shovqin, xona reverb/echo, musiqa yoki bir nechta so'zlovchi bo'lsa — **AI ularning hammasini ham nusxalaydi**. Datasetda ekspressiv o'zgaruvchanlik bo'lmasa, klon ham monoton chiqadi. | Obuna (Creator+/Pro) |
| **SpeechData.ai** — https://www.speechdata.ai/datasets/english-usa | Tayyor litsenziyalangan korpus (ASR/TTS uchun ruxsat). | **$90 / audio soat**, hajmiy chegirmalar bilan |
| **Appen** — https://www.appen.com/speech-and-audio-training-data | Off-the-shelf + custom collection: so'zlovchilarni jalb qilish, spetsifikatsiya bo'yicha yozib olish, o'nlab/yuzlab tillarda. | Ishlab chiqarilgan soatiga; **kam tarqalgan til va batafsil spetsifikatsiyada narx oshadi** |
| **Voice123 Enterprise TTS Datasets** — https://voice123.com/pages/enterprise-tts-voice-datasets/ | Professional voice actor bilan maxsus TTS dataset (til, soat, label, format bo'yicha). | So'rov bo'yicha |
| **Shaip** | SOC2 / GDPR / HIPAA mos, 70 000+ soat, 80+ mamlakat, studiya sifati. | So'rov bo'yicha |
| **LXT** — https://www.lxt.ai/services/post-editing-of-asr/ | ASR chiqishini inson tomonidan tuzatish xizmati (aynan foydalanuvchining review muammosi uchun). | So'rov bo'yicha |

**O'zbek tili uchun aniq narx**: ma'lumot topilmadi. Umumiy pattern shuni ko'rsatadi: kam tarqalgan til → soatiga narx sezilarli yuqori.

**Xulosa**: 10 soat audio uchun tijorat yo'l ~$900+ (faqat tayyor korpus narxi bo'yicha) va o'zbek uchun mavjud emas → o'z tool'ini qurish iqtisodiy jihatdan to'g'ri qaror.

---

## 11. Amaliy hamjamiyat tajribasi — REAL xatolar va ularning sabablari

Bu bo'lim eng qimmatli. Har bir element aniq manbaga havola bilan.

### 11.1 rhasspy/piper Discussion #191 — "US English Female Dataset"
Havola: https://github.com/rhasspy/piper/discussions/191

StoryHack'ning real workflow'i: Audacity'da MP3'larni birlashtirish → "Analyze → Label Sounds" (≥0.6 s sukunat, min 3 s) → 22050 Hz WAV eksport → PHP + Sox bilan >15 s yozuvlarni filtrlash → Whisper transkripsiya → `metadata.csv`.

**Hisobot qilingan nosozliklar va iqtiboslar:**
- ZachB100: **"Whisper completely missed most of the text in the first file, and likely others besides."** → Whisper birinchi faylning matnining ko'p qismini butunlay o'tkazib yuborgan. Bu avtomatik transkripsiyaga ishonishning bevosita narxi; **qo'lda tekshirish majburiy bo'ldi**.
- ZachB100: **"Right now, nothing about this process is quick and easy other than the training."** → Butun jarayonda **trainingdan boshqa hech narsa tez va oson emas**. Ya'ni bottleneck GPU emas, **dataset tayyorlash**.
- StoryHack keyinchalik avtomatik yo'ldan voz kechib, **PixViz bilan closed-caption yasab, qo'lda matn-audio hizalashga** o'tdi.
- Threaddagi konsensus: *"the way that generates the best dataset is still manual marking and slicing up the recordings, using the original text to get correct text and natural pauses and phrasing"* → **eng yaxshi dataset hali ham qo'lda belgilash va kesish bilan, asl matnni ishlatib to'g'ri matn, tabiiy pauza va frazirovka olish orqali** chiqadi.

**Bu aynan foydalanuvchining loyihasining asosli sababi**: asl matnni ishlatish g'oyasi to'g'ri va hamjamiyat ham shuni tasdiqlaydi; muammo shundaki, buni **avtomatlashtirgan yaxshi tool yo'q**.

### 11.2 coqui-ai/TTS Discussion #2507 — "Best Procedure For Voice Cloning"
Havola: https://github.com/coqui-ai/TTS/discussions/2507

Workflow: 15–25 daqiqa audio (biri augmentatsiya bilan 56 daqiqaga chiqargan) → **RNNoise** shovqin filtri → **Whisper** transkripsiya → 22 050 Hz ga downsample → 1–10 s klip.

| Kuzatilgan nosozlik | Aniqlangan sabab |
|---|---|
| **"Chipmunk" (burunduq) ovoz** | Dataset sample rate va `BaseAudioConfig`dagi e'lon qilingan sample rate **mos kelmasligi** |
| Robot / uzuq-yuluq ovoz | Juda erta checkpoint; **50k+ qadamdan keyin** tuzaldi |
| YourTTS 13k qadamda tushunarsiz | Model VITS'dan ko'ra uzoqroq training talab qiladi |
| Training uzilishi | Google Colab uzoq training vaqtida Drive'ni yo'qotib qo'yadi ("170 epoch maksimum") |
| ~50k qadamdan **keyin sifat pasaydi** | Overfitting |

Vaqt: VITS fine-tune batch 16 da **~4 soat/epoch**. Ishtirokchining aytgan asosiy to'sig'i: **hisoblash emas, hujjatlar yetishmasligi** — *"YourTTS ... does not have as much documentation ... I had to patch together something from hours of internet research."*

### 11.3 metadata.csv format tuzoqlari
- LJSpeech'da `metadata.csv` dagi transkript **birinchi kolonkadan keyin yana ikki kolonkaga nusxalangan** (`|` ajratgichi bilan). Bir xil struktura bo'lmasa, **"a lot of time troubleshooting"** sarflanadi. Manba: mozilla/TTS wiki — https://github.com/mozilla/TTS/wiki/Dataset
- Ya'ni tool `wav_id|raw_text|normalized_text` shaklini **default holda** chiqarishi kerak, aks holda foydalanuvchi keyingi bosqichda debug qiladi.

### 11.4 Klip uzunligi balansi (Coqui wiki)
Havola: https://github.com/coqui-ai/TTS/wiki/What-makes-a-good-TTS-dataset
- Faqat **qisqa klip (1–3 s)** bo'lsa → inference'da uzun gaplarda model qoqiladi.
- Klip **juda uzun** bo'lsa → model alignment'ni **umuman o'rganmaydi** yoki training juda cho'ziladi.
- Uzun va qisqa orasida balans kerak — bias'ni oldini olish uchun.
- Fon shovqini alignment o'rganishni qiyinlashtiradi → sifat past chiqadi.
- **Amaliy natija**: tool "min/max segment davomiyligi" va "uzunlik gistogrammasi" ko'rsatishi shart, faqat o'rtachani emas.

### 11.5 Whisper sukunatda gallyutsinatsiya qiladi — datasetni buzadigan eng sinsiz xato
Havola: https://github.com/openai/whisper/discussions/1606
- Whisper YouTube subtitrlarida o'qitilgan → **sukunat yoki past energiyali qismlarda o'qitish ma'lumotidan fraza "to'qib chiqaradi"** ("Thank you for watching!" kabi).
- Whisper'ning **ichki filtri ishonchsiz**: `no_speech_prob` (nutq yo'q token ehtimoli) va `avg_logprob` chegaralariga tayanadi. Lekin gallyutsinatsiyalangan chiqish ko'pincha **yuqori ishonch** bilan generatsiya qilinadi → `avg_logprob` yuqori, `no_speech_prob` esa kutilmaganda **past** → gallyutsinatsiya filtrdan o'tib ketadi.
- Ishlaydigan yechimlar:
  - **VAD / sukunat gate** (Silero VAD yoki webrtcvad) — faqat nutq bor bo'lakni Whisper'ga berish; 5 s bo'lakda nutq frame'i bo'lmasa bo'sh transkript qaytarish
  - `condition_on_previous_text=False` — bitta gallyutsinatsiya keyingi oynani "urug'lantirmasligi" uchun
  - **Takror (looping) detektsiyasi** — oldingi fragmentning qayta takrorlanishi
  - Energiya tekshiruvi: o'z trekida **peak < ~−60 dBFS** bo'lgan so'zni tashlash
- **Nega bu foydalanuvchi uchun kamroq xavfli**: unda ground-truth matn bor → gallyutsinatsiya datasetga tushmaydi. **Lekin** hizalash bosqichida Whisper/ASR ishlatilsa (fuzzy matching, chegara aniqlash uchun), gallyutsinatsiya **hizalashni siljitishi** mumkin → hizalashni ASR'dan mustaqil confidence bilan tekshirish kerak.

### 11.6 Whisper word-level timestamp'lari aniq emas
- Whisper **word-level timestamp uchun o'qitilmagan**; hozirgi chiqish — inference vaqtidagi "trik" natijasi, **ayniqsa pauzalarda** aniq bo'lmaydi. Manba: https://github.com/openai/whisper/discussions/1855
- Whisper ba'zan audio uzluksiz bo'lsa ham "pauza" (so'z timestamp'lari orasida bo'shliq) hisobot qiladi → **so'z o'rtasida bo'linish** kelib chiqadi.
- Audio so'z o'rtasida kesilsa, Whisper **tugash timestamp'ini umuman prognoz qilmasligi** mumkin.
- Fine-tune qilingan Whisper modellarida timestamp prognozi buziladi: https://github.com/linto-ai/whisper-timestamped/issues/173 — **o'zbek uchun fine-tune Whisper ishlatiladigan bo'lsa buni albatta sinash kerak.**
- **Xulosa**: Whisper timestamp'lari ustiga to'g'ridan-to'g'ri kesish qilish xato. Ustidan **forced alignment** (wav2vec2 / CTC) yugurtirish kerak — easyaligner/WhisperX/NFA aynan shuni qiladi.

### 11.7 Bo'laklab (chunked) parallellashtirishning narxi
Storyteller "Turbo Mode" hujjati (https://storyteller-platform.dev/docs/managing/aligning): boblarni kichik bo'laklarga bo'lib parallel ishlash sezilarli tezlik beradi, **lekin**:
- Bo'lak chegaralarida **aniqlik pasayadi** (timing offset)
- Chegarada **matn o'tkazib yuboriladi**
- **Qisqa boblar butunlay tashlab ketiladi**
- Tuzatish: turbo'ni `1` ga qaytarib **qaytadan hizalash**

Bundan tashqari: transkripsiya default holda **mavjud barcha CPU yadro va butun RAM'ni** oladi → Docker'da `--memory=24g --cpus=8` kabi cheklov kerak. Kitob boshiga **1–4 soat**.

### 11.8 ozen-toolkit issue'laridan UX darslari
https://github.com/devilismyfriend/ozen-toolkit/issues — 17 ochiq issue'dan asosiylari: dependency to'qnashuvi ("Installation failure: conflicts"), **til parametri qayerda?**, **kesish oralig'i parametri qayerda?**, **generatsiya qilingan audio uzunligini qanday nazorat qilaman?**, umumiy tushunarsiz xatolar, hujjat yetishmasligi.

→ **Yangi tool'da birinchi ekranda ko'rinishi kerak**: til, min/max segment davomiyligi, sukunat chegarasi, chiqish sample rate. Va **xato xabarlari aniq** bo'lishi kerak.

### 11.9 Windows-ga xos aniq to'siqlar (bir joyga to'plangan)
| To'siq | Manba |
|---|---|
| afaligner: **Windows qo'llanmaydi**, VM tavsiya | afaligner README |
| aeneas: Windows'da **xotira muammosi tufayli uzun kesimlar qilinmaydi** | youmebangbang/TTS-dataset-tools README |
| speech-dataset-generator: **faqat Ubuntu 22 test qilingan** | repo README |
| MFA: conda-forge orqali, rasmiy **"Troubleshooting on Windows"** bo'limi mavjud | https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html |
| Label Studio: Windows'da **Gohlke wheel**'lari kerak bo'lishi mumkin | Label Studio README |
| piper-recording-studio: ko'rsatmalar **Linux/Unix** uchun | repo README |
| **Yechim**: Purfview/whisper-standalone-win — barcha kutubxonalarni o'z ichiga olgan `.exe`, Python kerak emas | https://github.com/Purfview/whisper-standalone-win |

### 11.10 Dataset validatsiya bayroqlari ro'yxati (foydali checklist)
`Sandeepsrinivasan-14/voxtune` (https://github.com/Sandeepsrinivasan-14/voxtune — **0 ⭐, 2 commit: jiddiy loyiha emas**, lekin validatsiya ro'yxati foydali) datasetda bayroqlaydigan holatlar:
`missing` / `corrupted` / `silent` / `too short` / `too long` / `noisy` / `clipped` / `low sample rate` / `empty transcript` / `mismatched transcript` / `duplicate`.

Yangi tool ham aynan shu 11 holatni segment darajasida bayroqlashi kerak.

---

## 12. Arxitektura patternlari (kuzatilgan real dizaynlardan)

### 12.1 CLI vs GUI vs notebook — kim nima tanlagan
| Interfeys | Kim | Baho |
|---|---|---|
| **`.bat` drag-and-drop** | ozen-toolkit (`Drag_Here.bat`) | Windows foydalanuvchisi uchun eng past kirish bo'sag'asi. Lekin parametrlarni yashiradi → issue'lar "kesish parametri qayerda?" |
| **Gradio web UI** | Applio, RVC-WebUI | Voice AI hamjamiyatida de-facto standart. Windows'da ishlaydi, o'rnatish oson. |
| **Lokal web UI + alohida CLI eksporter** | piper-recording-studio | **Eng toza ajratma**: UI faqat yozib olish/ko'rish, `python -m export_dataset` alohida. Sinovga qulay. |
| **Docker web servis + DB** | Storyteller | Uzun ishlar (1–4 soat) uchun to'g'ri. Resurs cheklovlari aniq. |
| **YAML config + CLI** | NeMo SDP | Reproducible, versiyalanadi, review qilinadi. |
| **Python API + CLI** | Lhotse, timething | Kutubxona sifatida embed qilish oson. |
| **Kaldi-uslubidagi `run.sh --stage N`** | espnet, icefall | 20 yillik akademik konvensiya; resume bepul. |
| **Jupyter notebook** | Coqui `AnalyzeDataset`, `CheckSpectrograms` | Faqat **tahlil/vizualizatsiya** uchun. Pipeline uchun emas. |
| **GUI tab'lar + CLI ekvivalenti** | speech-splitter | Har GUI sozlamasining CLI muqobili bor → avtomatlashtirish mumkin. **Ideal.** |
| **Sof brauzer (Web Audio API)** | voice-studio, SpeechDataBuilder | Review uchun juda qulay, lekin **10 soatli fayl brauzer xotirasida muammo**. |

**Tavsiya etiladigan g'olib pattern**: **CLI/kutubxona yadro (barcha og'ir ishlov) + yupqa lokal web UI (faqat review va triage)**. speech-splitter va piper-recording-studio ikkisi ham shu tomonga boradi.

### 12.2 Checkpoint / resume — kuzatilgan usullar
1. **Manifest-per-stage** (NeMo SDP): har processor JSON-lines manifest o'qiydi va **yangi manifest yozadi**. Bosqich tugadi = fayl mavjud. Qayta ishga tushirish = mavjud faylni o'tkazib yuborish. **Eng oddiy va eng ishonchli.**
2. **Raqamlangan stage** (icefall/espnet): `--stage N --stop-stage M`. Foydalanuvchi qaysi bosqichdan davom etishni o'zi tanlaydi.
3. **Lazy manifest** (Lhotse): `jsonl.gz` — oqim sifatida o'qiladi, xotiraga to'liq yuklanmaydi; Shar shard'lari parallel ishga bo'linadi.
4. **Servis + per-ish status** (Storyteller): DB'da har kitobning holati; UI'da qayta ishga tushirish tugmasi.

**Tavsiya**: JSONL manifest zanjiri (NeMo uslubi) **kanonik artefakt** bo'lsin + har bosqich uchun `{stage}.done` yoki `state.json`. Kirish **content hash**i (audio fayl hash + matn fayl hash + parametr hash) saqlansa, o'zgarmagan bosqich avtomatik o'tkazib yuboriladi.

### 12.3 Oraliq holatni saqlash
- **JSONL manifest** (`audio_filepath`, `text`, `duration`, + hizalash maydonlari) — append-only, oqim bilan ishlaydi, `grep`/`jq` bilan tekshiriladi, git'ga diff qilinadi. **Kanonik format sifatida eng yaxshi.**
- **SQLite** — review UI uchun qulayroq: segment bo'yicha query, `status` (pending/accepted/rejected/edited) maydonini update qilish, filtrlash, tartiblash (confidence bo'yicha).
- **Eng yaxshi yechim: ikkisi ham** — JSONL asosiy artefakt (eksport/reproducibility), SQLite esa review indeksi (JSONL'dan tiklanadigan).
- **Sxema uchun tayyor namuna**: Lhotse `SupervisionSegment` — `recording_id`, `start`, `duration`, `channel`, `text`, `speaker`, `language`, `alignment`. Bu sinovdan o'tgan, uni qaytadan o'ylab chiqish shart emas.

### 12.4 10 soatli audioni RAM'ni tugatmasdan ishlash
**Muammo hisobi**: 10 soat × 3600 s × 22 050 Hz × 4 bayt (float32) mono ≈ **3.2 GB**. 48 kHz stereo bo'lsa ≈ **14 GB**. `librosa.load()` butun faylni float32 massivga yuklaydi → **bu xavfli va oldini olinishi kerak**.

Ishlaydigan yondashuvlar:
1. **`soundfile.blocks()`** — blok-blok o'qish, faqat bir necha blok xotirada bo'ladi: `sf.blocks('file.wav', blocksize=1024, overlap=512)`. Hujjat: https://python-soundfile.readthedocs.io/ — *"holds only a given number of audio blocks in memory and is therefore able to play files that are larger than the available RAM"*. `overlap` parametri chegara hodisalarini yo'qotmaslik uchun.
2. **`soundfile.SoundFile` + `.seek()` / `.read(frames)`** — random access; segmentni faqat kerak bo'lganda o'qish.
3. **NFA buffered streaming** — `chunk_len_in_secs` (default 1.6) + `total_buffer_in_secs` (default 4.0). NFA hujjati aniq: *"can align long audio files without running out of memory"*.
4. **easyaligner**: VAD → wav2vec2 emissions → Viterbi. Emissions matritsasi audio uzunligiga chiziqli o'sadi (raw audio'dan ~320× kichik) → soatlab audio bir o'tishda sig'adi. **Bo'laklash kerak emas** → chegara xatolari yo'q.
5. **Segmentni yozishda ffmpeg `-ss`/`-t`** bilan **asl fayldan to'g'ridan-to'g'ri** kesish — dekodlangan massivni Python xotirasida saqlamasdan. Bu eng kam RAM ishlatadigan usul va bit-exact.
6. **Praat `LongSound`** — vizual ko'rish/annotatsiya uchun diskdan o'qish (agar review'da uzun faylni to'liq ko'rsatish kerak bo'lsa, shu pattern'ni ko'chirish mumkin).
7. **Review UI'da**: butun waveform'ni brauzerga yubormaslik. Faqat (a) oldindan hisoblangan **past-rezolyutsiyali peak envelope** (masalan sekundiga 100 nuqta → 10 soat = 3.6M nuqta, bu ham juda ko'p → bob/bo'lim bo'yicha lazy yuklash) va (b) har segment uchun **alohida kesilgan kichik audio fayl**. voice-studio brauzerda hammasini qilishga urinadi — bu 10 soatda buziladi.

### 12.5 Uzun ishlarni progress va bekor qilish
- Storyteller darsi: uzun ish **default holda barcha resursni yeydi** → CPU yadro soni / RAM cheklovi **konfiguratsiya parametri** bo'lishi kerak.
- faster-whisper / WhisperX GPU'da **10–30× realtime** (speech-splitter o'lchovi), CPU'da **1–3×** → 10 soat audio: GPU'da ~20–60 daqiqa, CPU'da **3–10 soat**. Progress bar va **bekor qilish/davom etish** majburiy.

---

## 13. XULOSA: nimani QAYTA ISHLATISH, nimani O'ZI YOZISH

### QAYTA ISHLATISH (yozmang)
| Komponent | Nima olinadi |
|---|---|
| **Forced alignment yadrosi** | **kb-labb/easyaligner** (https://github.com/kb-labb/easyaligner) — MIT, tirik, soatlab audio bir o'tishda, confidence ball beradi, MMS bilan ko'p tilli. Muqobil/zaxira: **NVIDIA NFA** (Apache-2.0, 1+ soat OOM'siz) yoki **jianfch/stable-ts** `align()` + `refine()` (arxivlangan, lekin fork qilish mumkin). |
| **Segment chegarasini nozik sozlash** | **stable-ts** `refine()` va `regroup()` mantig'i (reproducible string konfiguratsiyasi) |
| **ASR (fuzzy matching / tekshirish uchun)** | **faster-whisper** yoki Windows'da **Purfview/whisper-standalone-win** (`.exe`, Python kerak emas) |
| **Sukunat/RMS kesish** | **openvpi/audio-slicer** algoritmi (MIT, 885 ⭐, kichik va aniq kod) |
| **Musiqa/shovqin filtri** | **YAMNet** (onnxruntime) — speech-splitter pattern'i |
| **Denoise / dereverb** | **DeepFilterNet** (speech-dataset-generator ishlatadi), yoki Applio'dagi UVR 'HP2-all-vocals' |
| **Sifat metrikalari ro'yxati** | **huggingface/dataspeech** metrikalari (SNR, C50, PESQ, SI-SDR, speaking rate, pitch std) — kodini emas, ro'yxatini |
| **Pipeline arxitekturasi** | **NeMo SDP** manifest-per-stage pattern'i + **icefall/espnet** `--stage` konvensiyasi |
| **Ma'lumot sxemasi** | **Lhotse** `SupervisionSegment` maydonlari |
| **Dataset tekshirish** | **idiap/coqui-ai-TTS**: `find_unique_chars.py`, `compute_statistics.py`, `remove_silence_using_vad.py`, `resample.py` + `AnalyzeDataset` / `CheckSpectrograms` notebooklari |
| **Chiqish format validatsiyasi** | `idiap/coqui-ai-TTS` `formatters.py` `ljspeech` funksiyasi bilan o'z chiqishini o'qib ko'rish (regression test) |
| **UX referens** | **piper-recording-studio** (lokal web UI + CLI eksporter ajratmasi), **speech-splitter** (GUI tab + CLI ekvivalenti), **voice-studio** funksiya ro'yxati (loop playback, zero-crossing split, micro-fade) |
| **Validatsiya bayroqlari** | voxtune ro'yxati: missing/corrupted/silent/too short/too long/noisy/clipped/low SR/empty/mismatched/duplicate |

### O'ZI YOZISH (bu qismda mavjud yechim yo'q)
1. **"Hizalash natijasi → LJSpeech segmentlari" konvertori** — TTS talablari bilan: 1–15 s oraliq, gap/punktuatsiya chegaralarida, so'z o'rtasida hech qachon kesmaslik, pauzaga kiradigan lekin qo'shni so'zga kirmaydigan bufer, `wav_id|raw_text|normalized_text` chiqishi, uzunlik gistogrammasi. **Hech bir mavjud tool buni to'liq va ishonchli qilmaydi.**
2. **Review/triage UI** — confidence bo'yicha tartiblangan, klaviatura-birinchi (Space=eshitish, Enter=qabul, E=tahrir, R=rad), faqat shubhali 10–15% ni ko'rsatadigan. Label Studio'ni sozlash mumkin, lekin og'ir; audino litsenziyasi noaniq; voice-studio prototip. **Bu eng katta differentsiator.**
3. **Resume/checkpoint qatlami** — content hash asosida, 10 soatli ishni yarim yo'ldan davom ettirish.
4. **Windows-native o'rnatish** — bir fayl/bir buyruq. Mavjud vositalarning yarmi Windows'da sinadi; bu tool'ning asosiy afzalligi bo'lishi mumkin.
5. **Xotira-xavfsiz audio I/O** — `soundfile.blocks()` + segmentni ffmpeg `-ss/-t` bilan asl fayldan kesish.

### ENG KATTA AMALIY TUZOQ (boshqalar hisobot qilgan)
**Avtomatik transkripsiyaga ishonish.** Piper thread'ida ZachB100 to'g'ridan-to'g'ri: *"Whisper completely missed most of the text in the first file"* — va butun jamoa xulosasi: *"nothing about this process is quick and easy other than the training"*. Whisper sukunatda **yuqori ishonch bilan** matn to'qib chiqaradi (`no_speech_prob` past, `avg_logprob` yuqori → ichki filtr uni ushlamaydi) va word-level timestamp'lari pauzalarda ishonchsiz, so'z o'rtasida bo'linish yaratadi.

Foydalanuvchining afzalligi shundaki, **unda haqiqiy matn bor** — bu tuzoqni yarim hal qiladi. Lekin qolgan yarmi baribir kuchli: **hizalash chegaralarining to'g'riligi hech qanday avtomatik metrika bilan to'liq kafolatlanmaydi**. Shuning uchun tool'ning markaziy g'oyasi "hizalashni mukammal qilish" emas, balki **"qaysi segmentga ishonmaslik kerakligini aniq ko'rsatish"** bo'lishi kerak — confidence ball, ASR-vs-reference farqi (NeMo SDP `ASRInference` pattern'i), chegara energiyasi tekshiruvi, davomiylik outlier'lari. Keyin inson faqat shubhali 10–15% ni ko'radi: 10 soat audio uchun ~1–3 soat, hammani eshitish uchun kerak bo'ladigan 10+ soat o'rniga.

---

## 14. Tasdiqlanmagan / topilmagan ma'lumotlar (halol ro'yxat)

- `youmebangbang/TTS-dataset-tools` oxirgi commit sanasi — **ma'lumot topilmadi**
- `gokhaneraslan/tts-dataset-generator` oxirgi commit sanasi — **ma'lumot topilmadi**
- `JarodMica/audiosplitter_whisper` litsenziyasi, yulduz soni, oxirgi commit — **ma'lumot topilmadi**
- `sushant-t/tts-trainer` metrikalari — **ma'lumot topilmadi**
- `linto-ai/whisper-timestamped` yulduz/oxirgi commit — **ma'lumot topilmadi**
- `easyaligner` Windows'da rasmiy test qilinganligi — **ma'lumot topilmadi** (pip + PyTorch bo'lgani uchun ishlashi kutiladi, lekin sinash kerak)
- `easyaligner` benchmark tezlik raqamlari va failure mode'lari — **hujjatlashtirilmagan**
- Label Studio OSS vs Enterprise audio UI farqining aniq texnik chegarasi — **aniq hujjatlashtirilmagan** (faqat Enterprise'da "advanced ... at scale" deyilgan)
- O'zbek tili uchun tijorat dataset narxi — **ma'lumot topilmadi**
- `midas-research/audino` aniq litsenziyasi (GitHub "NOASSERTION" ko'rsatadi) — **ishlatishdan oldin repo ichidagi LICENSE faylini tekshirish kerak**
- Reddit muhokamalari — WebSearch reddit.com ni indekslamaydi, shu sababli r/MachineLearning va r/LocalLLaMA thread'lari **tekshirilmadi**. Barcha hamjamiyat tajribasi GitHub discussions/issues va rasmiy hujjatlardan olingan.
- Coqui Discourse forumi (eski `discourse.mozilla.org` / Coqui forum) — **hozir onlayn topilmadi**; o'rniga GitHub Discussions ishlatildi.

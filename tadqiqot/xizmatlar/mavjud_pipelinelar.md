# Mavjud avtomatik TTS-dataset pipeline'lari: tadqiqot laboratoriyalari nima qurgan

**Tadqiqot sanasi:** 2026-10-01
**Savol:** "Nega hech kim avtomatik TTS-dataset tayyorlash tizimini qurmagan? Yoki qurganmi — qanday ishlaydi?"
**Qisqa javob:** Qurilgan. Kamida 14 ta nashr etilgan pipeline bor, 8 tasi kodi bilan. Muammo "yo'qligi"da emas — **qayta ishlatilishi**da, va **past resursli tilda ASR sifatiga bog'liqligi**da.

---

## 0. Umumiy jadval

| Pipeline | Bosqichlar (qisqa) | Kod ochiq? | Umumiy maqsadli? | Saqlanish darajasi | Havola |
|---|---|---|---|---|---|
| **Emilia-Pipe** (SLT 2024) | standartlashtirish → manba ajratish (UVR) → diarizatsiya (pyannote) → VAD segmentatsiya (Silero) → ASR (Whisper medium/WhisperX) → filtrlash (DNSMOS≥3.0) | ✅ Amphion ichida | ⚠️ ha, lekin faqat 6 til (zh/en/fr/ja/ko/de) | **29.43%** (598.87h → 176.22h, 600h namunada) | [arXiv:2407.05361](https://arxiv.org/abs/2407.05361), [kod](https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/README.md) |
| **WenetSpeech4TTS** (Interspeech 2024) | segment birlashtirish (gap<0.55s) → chegara kengaytirish (≤0.5s) → enhancement (MBTFNet) → ko'p-spiker aniqlash (ResNet293) → ASR (Paraformer-large) → sifat filtri + DNSMOS tier | ❌ kod relizi topilmadi | ❌ faqat WenetSpeech korpusiga | Basic 7,226h / 12,800h ≈ 56% (xom manbadan: ma'lumot topilmadi) | [arXiv:2406.05763](https://arxiv.org/abs/2406.05763), [HF](https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS) |
| **NeMo SDP** (NVIDIA) | YAML konfig: har bir `_target_` protsessor ketma-ket; manifest kirish/chiqish | ✅ Apache-2.0 | ✅ **ha — eng umumiy maqsadli** | konfiggan bog'liq | [GitHub](https://github.com/NVIDIA/NeMo-speech-data-processor), [docs](https://nvidia.github.io/NeMo-speech-data-processor/) |
| **Libriheavy** (ICASSP 2024) | locate (suffix array) → align (Levenshtein) → segment (2–30s, punktuatsiya chegarasida) | ✅ Apache-2.0 | ❌ `download/librilight` hardcoded; `text_search` kutubxonasi umumiy | ~83% (60,000h+ → 50,000h) | [arXiv:2309.08105](https://arxiv.org/abs/2309.08105), [kod](https://github.com/k2-fsa/libriheavy) |
| **IndicVoices-R** (NeurIPS 2024) | 44.1kHz filtri → HTDemucs → VoiceFixer → DeepFilterNet3 → C50/SNR/pitch/rate filtri → CER>5% filtri → volume norm | ❌ **pipeline kodi relizi yo'q** | ❌ | ma'lumot topilmadi (chiqish: 1,704h) | [arXiv:2409.05356](https://arxiv.org/abs/2409.05356), [repo](https://github.com/AI4Bharat/IndicVoices-R) |
| **GLOBE** (Interspeech 2024) | SNR<0dB / bandwidth<12kHz / sukunat>930ms filtri → Whisper + WFST norm, edit distance>1 filtri → spiker metadata tiklash | ❌ **pipeline kodi relizi yo'q** | ❌ Common Voice'ga xos | **~16%** (3,347h → 535h) | [arXiv:2406.14875](https://arxiv.org/abs/2406.14875), [HF](https://huggingface.co/datasets/MushanW/GLOBE) |
| **LibriTTS-R** (Interspeech 2023) | Miipher restoration: w2v-BERT → DF-Conformer (PnG-BERT shartli) → WaveFit-5 vokoder | ⚠️ dataset ochiq, **Miipher vaznlari relizi ko'rsatilmagan** | ❌ | **100%** (filtrlamaydi, tiklaydi: 585h → 585h) | [arXiv:2305.18802](https://arxiv.org/abs/2305.18802), [openslr.org/141](http://www.openslr.org/141/) |
| **YODAS** (ASRU 2023) | keyword/channel crawl → subtitr yuklab olish → evristik LID → CTC moslashtirish (threshold 2.0) | ⚠️ "ommaga ochiq bo'ladi" (va'da) | ⚠️ YouTube'ga xos | ma'lumot topilmadi | [arXiv:2406.00899](https://arxiv.org/abs/2406.00899) |
| **NVIDIA Granary** (Interspeech 2025) | audio prep (40s) → ikki o'tishli inference (LID→ASR) → LID tekshirish → gallyutsinatsiya filtri → char-rate → charset → P&C tiklash (Qwen2.5-7B) → AST (EuroLLM-9B) → CometKiwi filtri | ✅ NeMo-SDP ichida | ✅ **ha — JSON manifest bilan o'z audiongizni berasiz** | **60.7%** (~1M h → 643,237.57h); YODAS: **52.86%** | [arXiv:2505.13404](https://arxiv.org/abs/2505.13404), [konfig](https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/dataset_configs/multilingual/granary) |
| **HiFiTTS-2** (Interspeech 2025) | LibriVox API (48kHz) → sukunat kesish (50dB) → bandwidth baholash → bandwidth filtri (11/13kHz) → matn tiklash (87% kitobdan, 13% DistilBERT) → NFA + Parakeet alignment → Sortformer ko'p-spiker teg → ASR CER filtri | ✅ SDP'da `hifitts2` konfigi | ⚠️ LibriVox API'ga bog'langan | **82%** @22.05kHz, **71%** @44.1kHz (44.7k h → 36.7k / 31.7k h) | [arXiv:2506.04152](https://arxiv.org/abs/2506.04152), [HF](https://huggingface.co/datasets/nvidia/hifitts-2) |
| **EuroSpeech** (NeurIPS 2025 D&B) | manba yig'ish → media yuklab olish → transkript parsing (PDF/DOCX/HTML/SRT) → Silero VAD (3–20s) → Whisper Turbo → **ikki bosqichli CER alignment** → CER tier filtri | ⚠️ kod bor, **litsenziya "TBD"**, 4 commit | ✅ "source-agnostic" deb da'vo qilinadi | 78,128h → **78.2%** (CER<30%) / **65.4%** (CER<20%) / **41.0%** (CER<10%) | [arXiv:2510.00514](https://arxiv.org/abs/2510.00514), [kod](https://github.com/SamuelPfisterer/EuroSpeech) |
| **WorldSpeech** (2026) | standartlashtirish (24kHz mono, OCR) → Silero VAD (3–30s) + til-maxsus ASR → audio-transkript juftlash → sliding-window CER<0.3 → **iterativ qayta moslashtirish** | ⚠️ repo'da faqat `eval/`+`train/`; aligner EuroSpeech'da | ⚠️ qisman | ma'lumot topilmadi (chiqish: 65,072.4h) — **o'zbek: 33.7h** | [arXiv:2605.09167](https://arxiv.org/pdf/2605.09167), [HF](https://huggingface.co/datasets/disco-eth/WorldSpeech) |
| **MMS** (Meta 2023) | GPU Viterbi forced alignment (uroman + XLS-R) → oyat segmentatsiya (~12s) → ikkinchi model bilan qayta alignment (threshold −0.2) → CER>10% filtri | ✅ TorchAudio + fairseq | ✅ aligner umumiy (TorchAudio ichida) | 55K h → **44.7K h** (~81%); CER filtri ~**1.7%** olib tashladi | [arXiv:2305.13516](https://arxiv.org/abs/2305.13516), [kod](https://github.com/pytorch/fairseq/tree/master/examples/mms) |
| **Omnilingual ASR Corpus** (Meta 2025) | jamoa tomonidan yozilgan audio + kompensatsiyalangan mahalliy hamkorlar; 345+ til HF kartasida | ✅ modellar Apache-2.0, dataset CC-BY-4.0 | — (dataset, pipeline emas) | ma'lumot topilmadi | [arXiv:2511.09690](https://arxiv.org/pdf/2511.09690), [kod](https://github.com/facebookresearch/omnilingual-asr) |
| **OpenBibleTTS** (2026) | USFM/USX matn → timing-file (28 til) yoki ReadAlongs forced alignment (9 til) → pyannote diarizatsiya → 4 mezonli filtr | ✅ CC BY-SA 4.0 | ⚠️ Bibliya audio formatiga xos | ma'lumot topilmadi (chiqish: 3,469h / 37 til) | [arXiv:2606.09553](https://arxiv.org/pdf/2606.09553), [kod](https://github.com/McGill-NLP/open-bible-tts) |
| **LEMAS** (2026) | ingestion → MMS alignment (romanizatsiya) → ishonch-asosli filtr → JSON+MP3 | ⚠️ aniq URL ko'rsatilmagan | ⚠️ "language-agnostic" | ma'lumot topilmadi (chiqish: 150,144h / 10 til) | [arXiv:2601.04233](https://arxiv.org/abs/2601.04233) |
| **Lhotse** | **pipeline emas — infratuzilma.** Recording/SupervisionSegment/Cut/CutSet | ✅ Apache-2.0, `pip install lhotse` | ✅ 100+ korpus retsepti + o'z datasetingiz | — | [GitHub](https://github.com/lhotse-speech/lhotse), [docs](https://lhotse.readthedocs.io/en/latest/corpus.html) |
| **DataSpeech** (HuggingFace) | `main.py` (SNR/C50/pitch/rate/PESQ/SI-SDR) → `metadata_to_text.py` (binning) → `run_prompt_creation.py` (LLM tavsif) | ✅ MIT | ✅ har qanday HF dataset | — (filtrlamaydi, annotatsiya qiladi) | [GitHub](https://github.com/huggingface/dataspeech) |

---

## 1. Emilia-Pipe — "birinchi ochiq manbali in-the-wild preprocessing pipeline"

**Maqola:** *Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation* — He va boshqalar, **SLT 2024**, [arXiv:2407.05361](https://arxiv.org/abs/2407.05361) (v1 2024-07-07, v2 2024-07-13, v3 2024-09-07).
**Kengaytirilgan versiya:** *Emilia: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for Speech Generation* — [arXiv:2501.15907](https://arxiv.org/abs/2501.15907) (2025).

### 1.1 Oltita bosqich (aniq tartibda)

| № | Bosqich | Model / vosita | Parametr |
|---|---|---|---|
| 1 | **Standardization** | — | WAV, mono, **24 kHz**, 16-bit, **−20 dBFS** ga normalizatsiya |
| 2 | **Source Separation** | Ultimate Vocal Remover — `UVR-MDX-NET-Inst_HQ_3.onnx` | fon musiqa/shovqinni ajratish |
| 3 | **Speaker Diarization** | `pyannote/speaker-diarization-3.1` | HF token kerak |
| 4 | **Fine-grained Segmentation by VAD** | **Silero-VAD** | **3–30 sekund** segmentlar |
| 5 | **ASR** | **Whisper medium**, WhisperX + CTranslate2 optimizatsiyasi | 30s oynadan uzun emas |
| 6 | **Filtering** | til identifikatsiyasi + **DNSMOS P.835 OVRL** + outlier aniqlash | **OVRL ≥ 3.0** |

### 1.2 Saqlanish darajasi — va manbalar orasidagi FARQ

600 soatlik xom in-the-wild namunada ([arXiv:2407.05361v1](https://arxiv.org/html/2407.05361v1) va [arXiv:2501.15907v1](https://arxiv.org/html/2501.15907v1) da bir xil jadval):

| Holat | Soat | % |
|---|---|---|
| Xom kirish | 598.87 | 100.00% |
| Qayta ishlangan, filtrlanmagan | 340.54 | **56.86%** |
| Filtrlangandan keyin | 176.22 | **29.43%** |

**Sifat o'sishi:** xom DNSMOS 2.50 ± 0.62 → filtrlangan **3.26 ± 0.14**.

⚠️ **MUHIM NOMUTANOSIBLIK:** `29.43%` raqami — **faqat ~600 soatlik namunada** va **DNSMOS ≥ 3.0** chegarasida o'lchangan. 101k soatlik to'liq Emilia uchun xom kirish soatlari **hech qayerda e'lon qilinmagan**. Bundan tashqari, [arXiv:2501.15907](https://arxiv.org/html/2501.15907v1) ga ko'ra **Emilia-Large uchun DNSMOS chegarasi 3.0 dan 2.4 ga tushirilgan** — ya'ni 216k soatlik versiyaning saqlanish darajasi 29.43% dan sezilarli yuqori bo'lishi kerak, lekin **u raqam e'lon qilinmagan (ma'lumot topilmadi)**. Yagona o'zgarish shu bo'lgan: "the only modification was changing the DNSMOS filtering threshold" — yangi bosqich qo'shilmagan.

### 1.3 Hajm va litsenziya

| | Emilia | Emilia-YODAS | Emilia-Large |
|---|---|---|---|
| Jami | **101,654 h** | ~113.9k h | **216,313 h** |
| English | 46.8k | 92.2k | 139.0k |
| Chinese | 49.9k | 0.3k | 50.3k |
| German | 1.6k | 5.6k | 7.2k |
| French | 1.4k | 7.4k | 8.8k |
| Japanese | 1.7k | 1.1k | 2.8k |
| Korean | 0.2k | 7.3k | 7.5k |
| Litsenziya | **CC BY-NC-4.0** | **CC BY-4.0** | aralash |

Manba: [amphion/Emilia-Dataset HF kartasi](https://huggingface.co/datasets/amphion/Emilia-Dataset).

### 1.4 Begona odam o'z audiosini bera oladimi?

**Ha.** `config.json` dagi `"input_folder_path"` ni o'zgartirish kifoya; natija `input_folder_path_processed` ga yoziladi. Boshqa kalitlar: `"SAMPLE_RATE"` (24000), `separate.model_path`, `mos_model.primary_model_path`, `huggingface_token`, `language.supported` = `["zh","en","fr","ja","ko","de"]`.

⚠️ **O'zbek tili qo'llab-quvvatlanmaydi.** `language.supported` ro'yxatida yo'q, va 5-bosqichda Whisper medium ishlatiladi. O'zbekcha uchun 5- va 6-bosqichlarni almashtirish kerak.

**O'rnatish:** Python 3.9 + CUDA, `bash env.sh`, **ikkita onnx checkpointni qo'lda yuklab olish** (`UVR-MDX-NET-Inst_HQ_3.onnx`, `sig_bak_ovr.onnx`), pyannote uchun HF token. `export CUDA_VISIBLE_DEVICES`. Windows haqida alohida eslatma yo'q, lekin o'rnatish skripti bash.

**Tezlik:** 8× NVIDIA RTX 4090 da **daqiqada 2.50 soat** xom audio.

### 1.5 Nima qilmaydi (muallif tan oladi)

README'dagi TODO ro'yxati — bu **mahsulot roadmap'i emas, tugallanmaganlik tan olinishi**:
- pipeline tezligini optimallashtirish
- **4GB dan katta audioni qo'llab-quvvatlash**
- yaxshiroq shovqin bilan ishlash
- **ASR uzunligini 30 sekunddan oshirish**
- kengaytirilgan labelling

Maqoladagi cheklovlar ([arXiv:2501.15907](https://arxiv.org/html/2501.15907v1)):
1. **Diarizatsiya nomukammal** — "small proportion of speech segments containing more than one speaker or overlaps" qoladi.
2. **3–30 sekund tashqarisida ishlamaydi** — "generating speech outside this range may sometimes lead to unexpected outcome".

---

## 2. WenetSpeech4TTS — ASR korpusini TTS korpusiga aylantirish (xitoycha)

**Maqola:** *WenetSpeech4TTS: A 12,800-hour Mandarin TTS Corpus for Large Speech Generation Model Benchmark* — Ma va boshqalar, **Interspeech 2024**, [arXiv:2406.05763](https://arxiv.org/abs/2406.05763).

### 2.1 Bosqichlar (aniq tartibda, aniq chegaralar bilan)

1. **Adjacent Segments Merging** — ketma-ket segmentlar orasidagi interval **< 0.55 sekund** bo'lsa va **Resemblyzer** spiker o'xshashligi **> 0.65** bo'lsa birlashtiradi; gap uzunligi **20 sekund**ga yetguncha takrorlanadi.
2. **Boundary Extension** — segment boshini orqaga suradi, lekin oldingi qo'shni bilan o'rta nuqtadan oshmaydi, **maksimum 0.5 sekund**; oxirga ham xuddi shunday.
3. **Speech Enhancement** — **MBTFNet** modeli: shovqin, fon musiqa va boshqa aralashuvlarni olib tashlab "clean vocals" oladi.
4. **Multi-Speaker Detection** — **ResNet293** spiker embedding modeli (VoxCeleb'da o'qitilgan); segment **1.5s oyna / 0.75s qadam** bilan chunklanadi; spektral klasterlash, kosinus o'xshashlik **> 0.75** bo'lsa klasterlar birlashtiriladi; embeddinglar turli klasterlarga tushsa yoki **yarmidan ko'pi "centrifugal"** belgilansa — segment belgilanadi.
5. **Speech Recognition** — **Paraformer-large** (o'zi **6.74% CER** ko'rsatadi) bilan qayta transkripsiya + ishonch bali.
6. **Quality Filtering** — ko'p-spikerli segmentlar olib tashlanadi; **ASR ishonch bali < 0.7** bo'lganlar olib tashlanadi.

### 2.2 Uch pog'onali bo'linish (DNSMOS chegaralari)

| Subset | DNSMOS chegarasi | Soat |
|---|---|---|
| **Premium** | **≥ 4.0** | **945 h** |
| **Standard** | **≥ 3.8** | **4,056 h** |
| **Basic** | **≥ 3.6** | **7,226 h** |
| Rest | — | 5,574 h |
| **Jami** | — | **12,800 h** |

⚠️ Subsetlar **ichma-ich** (nested), ajralgan emas: "if you want to use the WenetSpeech4TTS Basic subset, you should download all the package files in the Premium, Standard, and Basic directories" — [HF kartasi](https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS).

### 2.3 Qayta ishlatish mumkinmi?

❌ **Yo'q.** Litsenziya CC-BY-4.0 lekin **faqat nokommersiya tadqiqot**; mualliflar audio mualliflik huquqiga ega emas. **Pipeline kodi relizi topilmadi (ma'lumot topilmadi)** — HF kartasida ham, maqolada ham GitHub havolasi yo'q. Bu to'liq **datasetni ulashish** maqolasi, vosita maqolasi emas.

### 2.4 Inson nazorati

Maqolada inson tekshiruvi bosqichi hujjatlashtirilmagan (ma'lumot topilmadi). Barcha filtrlar avtomatik.

---

## 3. NVIDIA NeMo Speech Data Processor (SDP) — eng umumiy maqsadli

**Repo:** [github.com/NVIDIA/NeMo-speech-data-processor](https://github.com/NVIDIA/NeMo-speech-data-processor) — **Apache-2.0**, Python 3.10+, `pip install -r requirements/main.txt` (+ ixtiyoriy NeMo). **221 star, 219 commit — faol qo'llab-quvvatlanadi.**
**Docs:** [nvidia.github.io/NeMo-speech-data-processor](https://nvidia.github.io/NeMo-speech-data-processor/)

### 3.1 Arxitektura: konfig-boshqaruvli YAML

Maqsad (hujjatdan): "Write code to process a new dataset, minimizing the amount of boilerplate code required" va "Share the steps for processing a speech dataset."

Har bir protsessor NeMo-uslubidagi **manifest** (JSONL) qabul qiladi, o'zgartiradi, manifest chiqaradi. YAML'da:
- `_target_` — protsessor klassi
- `output_manifest_file` — `${data_split}` kabi interpolyatsiya qo'llab-quvvatlanadi
- protsessorga xos parametrlar (regex, alifbo filtri, maydon tanlash)
- ixtiyoriy **`test_cases`** — validatsiya uchun (bu ishlab chiqarish darajasidagi dizayn belgisi)

Ishga tushirish:
```bash
python main.py --config-path="dataset_configs/<lang>/<dataset>/" \
  --config-name="config.yaml" workspace_dir=... processors_to_run="0:"
```

### 3.2 Mavjud dataset konfiglari

[dataset_configs/](https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/dataset_configs): `arabic`, `armenian`, `english`, `georgian` (MCV), `italian`, `kazakh`, `portuguese`, `spanish` (MLS), `spanish_pc`, **`uzbek`**, `commoncrawl`, `ipl`, `multilingual/granary`, `tts/ytc`.

🟢 **`dataset_configs/uzbek/` da uchta konfig bor:** `fleurs`, `mcv`, **`uzbekvoice`**. Bu foydalanuvchi uchun to'g'ridan-to'g'ri boshlang'ich nuqta.

### 3.3 `tts/ytc` — tayyor TTS pipeline konfigi

[config.yaml](https://raw.githubusercontent.com/NVIDIA/NeMo-speech-data-processor/main/dataset_configs/tts/ytc/config.yaml) protsessorlari aniq tartibda:

1. `CreateInitialManifestYTC` — 16 kHz mono WAV ga resampling
2. `PyAnnoteDiarizationAndOverlapDetection` — diarizatsiya + **overlap aniqlash**, maks **40s** segment
3. `SplitLongAudio` — **1 sekund** minimal pauza chegarasi
4. `NeMoASRAligner` — **Parakeet** modeli bilan matn/audio alignment, batch 16
5. `JoinSplitAudioMetadata`
6. `MergeAlignmentDiarization`
7. `InverseTextNormalizationProcessor` — tilga xos ITN
8. `TorchSquimObjectiveQualityMetricsProcessor` — obyektiv sifat metrikalari
9. `BandwidthEstimationProcessor` — bandwidth baholash
10. `PrepareTTSSegmentsProcessor` — yakuniy TTS segmentlari, tugatuvchi punktuatsiya bilan

### 3.4 Haqiqatan umumiy maqsadlimi? — HA

[Granary konfigi](https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/dataset_configs/multilingual/granary) **ixtiyoriy audioni qabul qiladi**. Faqat JSON manifest kerak:
```json
{"source_audio_filepath": "/path/to/file.flac"}
```
Ishga tushirish:
```bash
python ${SDP_DIR}/main.py \
  --config-path ${SDP_DIR}/dataset_configs/multilingual/granary/ \
  --config-name config.yaml \
  input_manifest_file=/path/to/input_manifest.json \
  output_dir=/path/to/output/dir sdp_dir=${SDP_DIR}
```
Sozlanadigan parametrlar: `source_lang`, `source_lang_full`, `translation.target_lang`, `min_audio_duration`, `max_audio_duration`, `min_audio_lid_probability`, `min_qe_score`, `max_len_diff_ratio`, `convert_to_audio_tarred_dataset.should_run`, `num_shards`. **14 ta hujjatlashtirilgan operatsiya.**

### 3.5 Nima qilmaydi / og'ir joylari

- ⚠️ **Windows rasman qo'llab-quvvatlanishi hujjatlashtirilmagan.** [NeMo Speech repo](https://github.com/NVIDIA-NeMo/NeMo) faqat Linux yo'llarini ko'rsatadi: `docker run ... nvcr.io/nvidia/nemo-speech:26.07.00`, `source .venv/bin/activate`. Native Windows o'rnatish ko'rsatmasi yo'q → amalda **WSL2 yoki Docker** kerak.
- Granary konfigi **Qwen2.5-7B-Instruct** (vLLM) va **CometKiwi** ni talab qiladi — bitta iste'molchi GPU'da og'ir.
- Minimal ASR o'rnatish: `uv pip install 'nemo-toolkit[asr]'`.

---

## 4. Libriheavy — 50k soat miqyosida uzun audio alignment

**Maqola:** *Libriheavy: a 50,000 hours ASR corpus with punctuation casing and context* — Kang, Yang, Yao, Kuang, Yang, Guo, Lin, Povey, **ICASSP 2024**, [arXiv:2309.08105](https://arxiv.org/abs/2309.08105).

### 4.1 Bosqichlar

1. **Locate** — Librispeech'da o'qitilgan ASR modeli 30 sekundlik **ustma-ust tushuvchi** segmentlarni transkripsiya qiladi; **suffix array** quriladi, maqsadli matnda "reverse-lexicographically close" pozitsiyalar topiladi → audio fayl kitobning qaysi qismiga to'g'ri kelishi aniqlanadi.
2. **Align** — **Levenshtein** masofasi, "longest increasing pairs" bilan aniqlangan bloklar ichida qo'llanadi.
3. **Segment** — **2–30 sekund** bo'laklar, **punktuatsiya chegaralarida** kesiladi + ball metrikalari.

### 4.2 Punktuatsiya va katta harf

🟢 **Model bilan tiklanmaydi — asl kitob matnidan olinadi.** Bu muhim dizayn tanlovi: agar sizda to'liq, to'g'ri yozilgan matn bo'lsa, punktuatsiya "bepul" keladi. (Foydalanuvchining "to'liq matn + uzun audio" g'oyasi aynan shu holat.)

### 4.3 Raqamlar

- Librilight **60,000+ soat** → Libriheavy **50,000 soat** ≈ **83% saqlanish**
- Subsetlar: small **500h**, medium **5,000h**, large **50,000h**
- Dev/test to'plamlarida train'dagi spiker va kitoblar bilan kesishma yo'q

### 4.4 Qayta ishlatish — ikki yarim javob

- ⚠️ **[k2-fsa/libriheavy](https://github.com/k2-fsa/libriheavy)** (Apache-2.0, 222 star, 14 commit) — audio katalogi **`download/librilight` ga hardcoded**. O'z audiongizni berish uchun hujjatlashtirilgan yo'l **yo'q**. Bosqichlar: `bash run.sh --stage X --stop-stage X`. Bog'liqliklar: **icefall, k2, lhotse, text_search**.
- ✅ **[k2-fsa/text_search](https://github.com/k2-fsa/text_search)** — `pip install fasttextsearch`. Bu alignment algoritmining **umumiy maqsadli** qismi: uzun audio ↔ uzun matn moslashtirish (suffix array + Levenshtein). Windows wheel mavjudligi hujjatlashtirilmagan (ma'lumot topilmadi); CMake build ko'rsatmalari bor. 79 star, 65 commit.

Mualliflar da'vosi: "we open-source our dataset creation pipeline **which can also be used to other audio alignment tasks**".

### 4.5 Cheklov

"requires a relatively long text to guarantee its accuracy" — qisqa matnlarda suffix-array moslashtirish ishonchsiz.

---

## 5. IndicVoices-R — ASR korpusini TTS korpusiga aylantirish (foydalanuvchi g'oyasiga eng yaqin)

**Maqola:** *IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS* — AI4Bharat, **NeurIPS 2024**, [arXiv:2409.05356](https://arxiv.org/abs/2409.05356).

### 5.1 Bosqichlar va aniq chegaralar

| № | Bosqich | Vosita / chegara |
|---|---|---|
| 1 | **Preprocessing** | faqat **44.1 kHz** audio saqlanadi (**8 kHz chiqarib tashlanadi**); **> 30 sekund** utterance'lar olib tashlanadi; mono → stereo upmix |
| 2 | **Demixing** | **HTDemucs** — manba ajratish + shovqin kamaytirish |
| 3 | **Dereverberation** | **VoiceFixer** — reverberatsiya va raqamli artefaktlarni kamaytirish |
| 4 | **Speech Enhancement** | **DeepFilterNet3** (full-band spektrogramma) |
| 5a | **Audio filtering** | **C50 ≥ 30 dB**; **SNR ≥ 25 dB**; davomiylik **0.2–30 s**; pitch o'rtacha **≤ 350 Hz**; pitch std **≤ 150 Hz**; gapirish tezligi **≤ 30 belgi/sekund** |
| 5b | **Transcript filtering** | transkript darajalari orasida **CER > 5%** bo'lsa chiqarib tashlanadi |
| 6 | **Post-processing** | **PyDub** bilan volume normalizatsiya, **0.1 dB headroom** |

### 5.2 Natija

**1,704 soat**, **10,496 spiker**, **22 til**, **689.6K utterance**. Litsenziya **CC-BY-4.0**.
**Saqlanish darajasi maqolada aniq ko'rsatilmagan — ma'lumot topilmadi.**

### 5.3 🔴 Eng muhim ogohlantirish: pipeline kodi RELIZ QILINMAGAN

[github.com/AI4Bharat/IndicVoices-R](https://github.com/AI4Bharat/IndicVoices-R) tarkibi: `VoiceCraft/` (model fine-tuning), `assets/`, `data_links.txt`, `README.md`, `LICENSE.md`. **Demixing / dereverberation / enhancement / filtrlash kodi yo'q.** Foydalanuvchi faqat tayyor datasetni oladi. O'z audiosiga qo'llash mexanizmi mavjud emas. 65 star, 19 commit, 9 open issue.

Ya'ni: **bosqichlar va chegaralar hujjatlashtirilgan, lekin implementatsiya yo'q.** Buni qaytadan yozish kerak — ammo barcha komponentlar (HTDemucs, VoiceFixer, DeepFilterNet3) o'z-o'zidan ochiq, shuning uchun qayta qurish amalda mumkin.

### 5.4 Cheklovlar (muallif tan oladi)

- "Could not unlock the potential of **conversational subset** of IndicVoices due to their low sampling rates of **8 kHz**" — ya'ni eng qimmatli, spontan qismi tashlab ketilgan.
- "dataset has not attained the caliber of **studio-recorded** speech datasets".

---

## 6. GLOBE — Common Voice'ni TTS korpusiga tozalash

**Maqola:** *GLOBE: A High-quality English Corpus with Global Accents for Zero-shot Speaker Adaptive Text-to-Speech* — Wang va boshqalar, **Interspeech 2024**, [arXiv:2406.14875](https://arxiv.org/abs/2406.14875).

### 6.1 Bosqichlar va chegaralar

**1-bosqich — namuna preprocessing + filtrlash (uch metrika):**
- **SNR < 0 dB** → chiqarib tashlanadi
- **signal bandwidth < 12 kHz** → chiqarib tashlanadi
- **> 930 millisekund** uzluksiz ichki sukunat → olib tashlanadi

**2-bosqich — nutq-matn moslashtirish:**
- Barcha utterance'lar **Whisper** bilan transkripsiya qilinadi
- **WFST** (weighted finite-state transducer) matn normalizatsiyasi
- **edit distance > 1** bo'lsa chiqarib tashlanadi
- ketma-ket so'z takrorlanishlari olib tashlanadi

**3-bosqich — spiker ma'lumotlarini tiklash:**
- **< 5 spiker yoki < 10 utterance** bo'lgan aksentlar birlashtiriladi
- ma'nosiz teglar olib tashlanadi
- 3 ta prediktor o'qitiladi: aksent **97.22%**, yosh **99.55%**, jins **99.95%** aniqlik
- nutq enhancement + VAD bilan sukunat kesish

### 6.2 Saqlanish — eng past

| | Common Voice | GLOBE | % |
|---|---|---|---|
| Soat | **3,347 h** | **535 h** | **~16%** |
| Spiker | **88,904** | **23,519** | ~26% |
| Aksent | — | **164** | — |

24 kHz, litsenziya **CC0-1.0** ([HF](https://huggingface.co/datasets/MushanW/GLOBE)).

### 6.3 Versiyalar

- **GLOBE** — 535 h, 24 kHz
- **[GLOBE_V2](https://huggingface.co/datasets/MushanW/GLOBE_V2)** — **44.1 kHz**; ⚠️ "**removed some samples (~5%) due to the volume and text alignment issues**" — ya'ni V1'da filtr o'tkazib yuborgan muammolar keyin qo'lda topilgan
- **[GLOBE_V3](https://huggingface.co/datasets/MushanW/GLOBE_V3)** — **724,000 qator**, 0.56–22.8 s; annotatsiyalar: **SNR** (−20…100), **UTMOS** (1.23…4.44), bashorat qilingan jins/yosh/aksent, Whisper Large v3 transkriptlari, asl `common_voice_sentence_id`; **234** aksent qiymati; CC0-1.0

### 6.4 🔴 Pipeline kodi RELIZ QILINMAGAN

Maqolada "We will release GLOBE publicly after acceptance" deyilgan, lekin **kod repozitoriysi havolasi yo'q**. HF kartasida ham tozalash kodi havolasi yo'q. Dataset ochiq (CC0), pipeline yopiq.

---

## 7. LibriTTS-R — mavjud korpusga nutq tiklash o'tishi

**Maqola:** *LibriTTS-R: A Restored Version of LibriTTS Corpus* — Koizumi va boshqalar (Google), **Interspeech 2023**, [arXiv:2305.18802](https://arxiv.org/abs/2305.18802).

### 7.1 Bu filtr emas — tiklash

**Miipher** — matn bilan xabardor parametrik re-sintez modeli, uchta komponent:
1. **w2v-BERT** — buzilgan to'lqin shaklidan o'z-o'zidan nazoratli (SSL) tasvirlar
2. **DF-Conformer feature-cleaner** — **PnG-BERT** lingvistik shartlashtirish bilan toza w2v-BERT xususiyatlarini bashorat qiladi
3. **WaveFit-5** neyron vokoder — tozalangan xususiyatlardan to'lqin sintez qiladi

Nishon: **"phoneme masking"** (shovqin/reverb nutqni yashiradi) va **"phoneme deletion"** (ishlov berishdan yo'qolgan chastota komponentlari).

### 7.2 Saqlanish: 100%

**585 soat, 2,456 spiker, 24 kHz** — asl LibriTTS bilan bir xil. **Hech bir utterance tashlanmagan.** Bu boshqa barcha pipeline'lardan tubdan farq qiladi: ular tashlaydi, bu tuzatadi.

Natija: 103-raqamli spiker uchun MOS **2.75 → 4.31 ± 0.08**.

### 7.3 🔴 Nosozlik va reliz muammosi

- "**two of 640 LibriTTS-R speech samples were distorted due to the failure of SR**" — ya'ni ~0.3% buzilgan va bu **qo'lda tinglash** orqali topilgan. Mualliflar train splitlarida aniqlanmagan nosozliklar borligini tan oladi.
- ⚠️ **Miipher kodi/vaznlari relizi maqolada ko'rsatilmagan (ma'lumot topilmadi).** Ya'ni eng qimmatli bosqich — eng kam takrorlanadigan bosqich. Dataset [openslr.org/141](http://www.openslr.org/141/) da LibriTTS bilan bir xil cheklovsiz litsenziyada mavjud, lekin boshqa korpusga qo'llash uchun model yo'q.

---

## 8. YODAS / YODAS-Granary / NVIDIA Granary — veb miqyosi va uning sifat muammolari

### 8.1 YODAS

**Maqola:** *YODAS: Youtube-Oriented Dataset for Audio and Speech* — ASRU 2023, [arXiv:2406.00899](https://arxiv.org/abs/2406.00899).

**Bosqichlar:**
1. **Keyword-based crawl client** — ko'p tilli **Wikipedia dump**'laridan so'rov ro'yxatlari tuziladi, kam vakillikdagi tillarga ustuvorlik; YouTube'ning o'z filtri Creative Commons litsenziya + subtitr mavjudligini talab qiladi; **AJAX crawling** bilan pastroq natijalarga chiqiladi
2. **Channel-based client** — topilgan videolardan kanallar ajratiladi (qidiruv mashhur videolarni takrorlaydi)
3. **Download client** — mono **24 kHz** ga aylantiradi, barcha subtitrlarni (qo'lda + avtomatik) oladi
4. **Evristik LID** — bitta noyob qo'lda subtitr bo'lsa to'g'ri deb qabul qilinadi; bir-biriga qarama-qarshi tillar bo'lsa "unlabeled" deb belgilanadi. ⚠️ Mualliflar: **"Language Identification (LID) tools" muvaffaqiyatsiz chiqdi** va bu kelajak ishi deb belgilangan
5. **Speech-text alignment** — oldindan o'qitilgan akustik model **CTC loss** bilan utterance'larni baholaydi; **alignment threshold 2.0** bilan nomos juftliklar filtrlanadi
6. Har bir til uchun maksimum **1,000,000** utterance train, **1,000** test

**Hajm:** **560,000+ soat** jami:
- Manual subset: **86,400 h**, **140 til**
- Automatic subset: **335,845 h**, 14–20 til
- Unlabeled: **144,174 h**
- Litsenziya: CC BY 3.0 (belgilangan subsetlar)

**🔴 Hujjatlashtirilgan sifat muammolari:**
- Automatic subset'da **"a significant proportion of misaligned utterances"**
- Nomos misollar musiqa/fon shovqini bilan **50 sekundga** cho'zilgan
- To'g'ridan-to'g'ri taqqoslash: manual subset'da o'qitish **14.9% CER**, automatic'da **32.3% CER**
- Automatic'da **o'chirish (deletion) xatosi 26.6**, manual'da **7.0** — ya'ni avtomatik subtitrlar nutqning katta qismini qoldirib ketadi
- Manual subset'da ham sifat mezoni: **< 10 sekund** va **alignment bali > 5**

**Kod:** "data collection software will be open to the public for individual use or collaborative efforts" — ⚠️ maqoladagi **va'da**, konkret repo havolasi berilmagan.

### 8.2 NVIDIA Granary — YODAS'ni tozalash

**Maqola:** *Granary: Speech Recognition and Translation Dataset in 25 European Languages* — Rao Koluguri va boshqalar (NVIDIA + CMU + FBK), **Interspeech 2025**, [arXiv:2505.13404](https://arxiv.org/abs/2505.13404).

**Manbalar va ularning xom hajmi:**
| Manba | Xom soat |
|---|---|
| YODAS | **363,549.3 h** |
| YouTube-Commons (YTC) | **255,333.72 h** |
| MOSEL (VoxPopuli + LibriLight) | **440,712.51 h** |

**To'qqiz bosqich (aniq tartibda):**
1. **Audio prep & segmentation** — FLAC/WAV, **16 kHz mono**, maks **40 sekund**; segmentatsiya: YODAS/YTC uchun **Whisper timestamp**'lari, LibriLight uchun **Parakeet TDT decoder** timestamp'lari
2. **Two-pass inference** — birinchi o'tish **til ID** bashorati, ikkinchi o'tish transkripsiya (bashorat qilingan LID metadata sifatida ishlatiladi); **FasterWhisper**, beam size **5**, **Silero VAD** (**400 ms** padding)
3. **LID verification** — Whisper bashorat qilgan LID maqsadli tilga mos kelmasa chiqarib tashlanadi; VoxPopuli uchun **`lid_prob < 0.8`** chiqarib tashlanadi
4. **Hallucination filtering** — uch belgi: takrorlanuvchi n-gramlar, uzun so'zlar, va **tilga xos qo'lda yozilgan "tez-tez gallyutsinatsiya qilinadigan iboralar" ro'yxati** (masalan fransuzcha "Sous-titrage Société Radio-Canada")
5. **Character-rate filtering** — tilga va korpusga xos evristikalar: g'ayritabiiy past/baland belgi tezligi
6. **Character set filtering** — **300+ belgi va simvoldan** iborat superset
7. **Punctuation & capitalization restoration** — **Qwen 2.5-7B-Instruct**; ⚠️ agar Qwen chiqishi Whisper transkriptidan **5% CER**dan ko'p farq qilsa, **asl pseudo-label saqlanadi**
8. **Translation generation** — **EuroLLM-9B**, greedy, vLLM inference (X→En AST)
9. **AST filtration** — length ratio filtri, belgi histogrammasi, **FastText LID**, **CometKiwi** sifat bahosi (`cometoid-wmt23`) — **NeMo-Curator** orqali

**Saqlanish darajasi:**
| | Xom | Chiqish | % |
|---|---|---|---|
| Umumiy ASR | ~**1,000,000 h** | **643,237.57 h** | **60.7%** |
| YODAS | 363,549.3 h | **192,172.16 h** | **52.86%** |
| AST | — | **351,048 h** | — |

**Natija da'vosi:** bu ma'lumotda o'qitilgan modellar **~50% kam ma'lumot** bilan bir xil ishlashga erishadi.

**Litsenziya:** dataset **CC-BY-4.0**, pipeline **NeMo-SDP** ichida Apache-2.0 ([HF](https://huggingface.co/datasets/nvidia/Granary), **76 til subseti** ASR+AST uchun).

### 8.3 YODAS-Granary

[espnet/yodas-granary](https://huggingface.co/datasets/espnet/yodas-granary) — Granary'ning `espnet/yodas2` ustida ishlatilgan **23 Evropa tili** uchun subsети. Pseudo-label: `Systran/faster-whisper-large-v3`; punktuatsiya/katta harf: `Qwen/Qwen2.5-7B-Instruct`. Pipeline — NeMo-SDP.

---

## 9. HiFiTTS-2 — audiokitobdan datasetga (foydalanuvchining asl g'oyasiga eng yaqin)

**Maqola:** *HiFiTTS-2: A Large-Scale High Bandwidth Speech Dataset* — Langman, Yang, Neekhara, Hussain, Casanova, Bakhturina, Li (NVIDIA), **Interspeech 2025**, [arXiv:2506.04152](https://arxiv.org/abs/2506.04152).

### 9.1 Bosqichlar va chegaralar

| № | Bosqich | Tafsilot |
|---|---|---|
| 1 | **LibriVox API download** | asl **48 kHz** fayllar (MLS'dagi 16 kHz qayta ishlangan audiodan EMAS) → **44.1 kHz** FLAC |
| 2 | **Silence trimming** | energiya asosida, chegara **50 dB**, bosh/oxirda maksimum **0.5 sekund** sukunat qoldiradi |
| 3 | **Bandwidth estimation** | har bir audiokitobning **birinchi 30 sekundi**da; quvvat spektri cho'qqidan **≥ −50 dB** bo'lgan maksimal chastota |
| 4 | **Bandwidth filtering** | 22.05 kHz subset: **≥ 11 kHz**; 44.1 kHz subset: **≥ 13 kHz** (24 kHz yoki pastroq yozuvlarni chiqarib tashlaydi) |
| 5 | **Text preprocessing** | **87%** asl audiokitob matnidan tiklangan; **13%** uchun **NeMo DistilBERT** punktuatsiya/katta harf bashorati; NeMo text normalization |
| 6 | **Forced alignment & segmentation** | **NeMo Forced Aligner** + **Parakeet-TDT-CTC**; utterance'lar **nuqtadan keyin ≥ 0.08 sekund** pauzada kesiladi |
| 7 | **Multi-speaker detection** | Sortformer modeli ~**104.4 soat**ni ko'p-spikerli deb aniqladi → ⚠️ **tuzatmaydi, faqat metadata sifatida teglaydi** |
| 8 | **ASR verification** | Parakeet-TDT-CTC WER/CER; **CER ≥ 100%** bo'lganlar filtrlanadi; qolgan metrikalar **metadata sifatida** qoldiriladi (foydalanuvchi o'zi filtrlashi uchun) |

### 9.2 Saqlanish darajasi

| | Soat | % | Utterance | Spiker |
|---|---|---|---|---|
| Kirish (MLS English) | ~**44.7k h** | 100% | — | — |
| **22.05 kHz chiqish** | **36.7k h** | **82%** | **13.1M** | **5,013** |
| **44.1 kHz chiqish** | **31.7k h** | **71%** | **11.3M** | **4,631** |

### 9.3 Qayta ishlatish va cheklovlar

- ✅ SDP hujjatlarida **`hifitts2` konfigi** ro'yxatlangan → pipeline NeMo-SDP (Apache-2.0) ichida.
- ⚠️ **LibriVox API'ga bog'langan** — boshqa audiokitob manbasi uchun 1-bosqichni qayta yozish kerak.
- 🔴 Cheklov: "mixed-bandwidth audio can be problematic for training speech models" va LibriVox'ning ko'p 44.1 kHz ma'lumotida eng yuqori chastota diapazonlarida spektral ma'lumot cheklangan.
- Dataset: [nvidia/hifitts-2](https://huggingface.co/datasets/nvidia/hifitts-2), to'liq **public domain** LibriVox'dan → kommersiya uchun yaroqli.

### 9.4 Boshqa audiokitob→dataset pipeline'lari

| Pipeline | Manba | Yondashuv |
|---|---|---|
| **Libriheavy** | Librilight (LibriVox) | suffix array + Levenshtein, kitob matni bilan (§4) |
| **OpenBibleTTS** (2026) | Bibliya audio, **37 til** | timing-file (28 til) yoki ReadAlongs forced alignment (9 til) — §12 |
| **WorldSpeech** | LibriVox, Aozora, Ben-Yehuda (CC0) + parlament/radio | sliding-window CER (§11) |
| **LEMAS** (2026) | 10 til, umumiy manbalar | MMS forced alignment + ishonch filtri |

---

## 10. Meta: MMS va Omnilingual ASR Corpus — 1000+ tilga qanday chiqdilar

### 10.1 MMS

**Maqola:** *Scaling Speech Technology to 1,000+ Languages* — Pratap va boshqalar (Meta), 2023, [arXiv:2305.13516](https://arxiv.org/abs/2305.13516).

**MMS-lab (juftlangan ma'lumot) — Yangi Ahd yozuvlari:**

| Qadam | Tafsilot |
|---|---|
| Manba | Faith Comes By Hearing, goto.bible, bible.com → **1,626 yozuv, 1,362 til, 55K soat** (MMS-lab-U) |
| **Boshlang'ich alignment** | **XLS-R** fine-tune: FLEURS + CommonVoice 8.0 (**8K soat, 127 til**); matn **`uroman`** transliteratsiya vositasi bilan romanizatsiya qilinadi (barcha yozuvlar uchun) |
| **GPU forced alignment** | Viterbi yo'lini **O(T×L) → O(L)** xotirada hisoblaydigan GPU versiyasi → **43 daqiqali** yozuvlarni qayta ishlash imkoni; audioda hisobga olinmagan kontent bo'lsa **star token ⟨*⟩** |
| **Yaxshilangan alignment** | ikkinchi model (**31K soat, 1,130 til**) bilan qayta moslashtirish; ball chegarasi **−0.2** (forced va greedy dekodlash normallashtirilgan ehtimollik farqidan) |
| **Segmentatsiya** | uzun bob yozuvlari **oyat**larga bo'linadi, o'rtacha **~12 sekund** |
| **Sifat filtri** | yarmida o'qitilgan monolingual ASR modellari qolganini filtrlaydi; **CER > 10%** bo'lganlar olib tashlanadi → **barcha namunalarning ~1.7%**; qo'shimcha **3,837 yozuv** (CER > 5%) butunlay olib tashlanadi |
| **Natija** | **1,107 til, 44.7K soat** (36.8K train / 3.5K dev / 4.4K test), o'rtacha **32 soat/til** |

**MMS-unlab (belgilanmagan):**
- Global Recordings Network: **6,255+ til** → **3,860 til, 9,345 soat**
- **inaSpeechSegmenter** bilan nutq/musiqa/shovqin segmentatsiyasi; nutq segmentlari oraliq nutqsiz segment **< 20%** davomiylikda bo'lsa birlashtiriladi; tasodifiy **5.5–30 sekund** bo'laklar
- **< 5 daqiqa** ma'lumotli 51 til olib tashlanadi → **3,809 til, 7.7K soat**

**✅ Qayta ishlatish:** GPU forced aligner **TorchAudio ichiga qo'shilgan** — ya'ni `pip install torchaudio` bilan ishlatish mumkin, bu **eng yaxshi "research → product" hikoyasi** bu ro'yxatda. Modellar va kod: [fairseq/examples/mms](https://github.com/pytorch/fairseq/tree/master/examples/mms).

### 10.2 Omnilingual ASR Corpus

**Maqola:** *Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages* — Meta, 2025, [arXiv:2511.09690](https://arxiv.org/pdf/2511.09690).

- **1,600+ til**, shundan **500+ til hech qachon biror ASR tizimida xizmat ko'rsatilmagan**
- Usul: ommaviy resurslar + **kompensatsiyalangan mahalliy hamkorliklar orqali jamoadan yozib olingan audio** — ya'ni **avtomatlashtirish emas, inson mehnatini sotib olish**
- 7B-LLM-ASR: **1,600+ til uchun 78%**ida **CER < 10**
- [facebook/omnilingual-asr-corpus](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus): HF kartasida **345+ til**, 100K–1M yozuv, **CC-BY-4.0**
- Modellar **Apache-2.0**, kod: [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr); ko'p tilli dataset tayyorlash uchun end-to-end qo'llanma, HF integratsiyasi, parquet ishlovi, CTC va LLM modellari uchun tayyor workflow'lar
- ⚠️ Jami soat va sifat nazorati qadamlari HF kartasida topilmadi — **ma'lumot topilmadi**

**Xulosa:** Meta 1000+ tilga **ikki xil yo'l** bilan chiqdi: (a) MMS — bitta parallel korpus (Bibliya) + arzon forced alignment; (b) Omnilingual — pul to'lab yangi ma'lumot yozdirish. **Hech biri "in-the-wild audioni avtomatik tozalash" emas.**

---

## 11. EuroSpeech va WorldSpeech — "source-agnostic" bo'lishga eng yaqin urinish

### 11.1 EuroSpeech

**Maqola:** *EuroSpeech: A Multilingual Speech Corpus* — Pfisterer, Grötschla va boshqalar (ETH Zürich DISCO), **NeurIPS 2025 Datasets & Benchmarks**, [arXiv:2510.00514](https://arxiv.org/abs/2510.00514).

**To'liq pipeline:**
1. **Data sourcing** — **22 Evropa parlament sayti qo'lda tekshiriladi**; audio/video URL va transkript havolalari standartlashtirilgan CSV'ga chiqariladi
2. **Media download** — format-agnostik handler'lar (to'g'ridan-to'g'ri havola, YouTube, dinamik sahifalar); PostgreSQL kuzatuv; checkpointing va xatolardan tiklanish
3. **Transcript preprocessing** — **PDF, DOCX, HTML, TXT, SRT** parserlari; ixtiyoriy **LLM tozalash (Gemini Flash 2.0)** nutqsiz elementlarni olib tashlash uchun
4. **VAD** — **Silero VAD**, **3–20 sekund** utterance'lar
5. **ASR** — **Whisper Turbo** (maltiy uchun tilga xos fine-tune qilingan Whisper)
6. **🟢 Ikki bosqichli alignment** — bu eng qimmatli algoritmik hissa:
   - **Coarse search:** sliding window bilan ASR matnini parlament transkriptiga solishtirish, har bir oyna uchun **CER** hisoblash; **CER < 30%** nomzodlar refinement'ga o'tadi; agar hech biri o'tmasa — **eng past CER'li 3 nomzod** o'tkaziladi
   - **Refined search:** boshlanish pozitsiyasini **±15 so'z**, oyna o'lchamini **L ± 15 so'z** (L = ASR segment uzunligi) o'zgartirib CER'ni minimallashtiradi; oldingi alignment xato joylashgan bo'lsa transkript boshidan **restart fallback**
7. **CER tier filtering** — uch sifat darajasi

**Saqlanish darajasi:**
| Daraja | Soat | % |
|---|---|---|
| Kirish (moslashtirilgan audio) | ~**78,128 h** | 100% |
| **CER < 30%** | **61,006 h** | **78.2%** |
| **CER < 20%** | **50,573 h** | **65.4%** |
| **CER < 10%** | **32,256 h** | **41.0%** |

**22 til**, barchasi **> 500 soat**, 19 tasi **> 1,000 soat**. Eng yuqori: Xorvatiya 5,616 h, Daniya 5,560 h, Norvegiya 3,867 h. Eng past: Malta 613 h, Islandiya 647 h, Bosniya 691 h.

**Tasdiqlash:** Whisper v3 Turbo'ni tilga **~200 soat** bilan fine-tune qilish 6 til (maltiy, island, litva, latish, sloven, eston) uchun **FLEURS** (domendan tashqari) test'da **o'rtacha 41.8% nisbiy WER kamayishi** berdi.

**🟢 Mualliflar da'vosi (to'g'ridan-to'g'ri):**
- "a **scalable, open-source pipeline** for constructing speech datasets from parliamentary proceedings"
- "specifically designed to be **robust against non-verbatim transcripts** and **easily adaptable to new data sources with minimal engineering effort**"
- "enable **non-expert teams to replicate this process** for new languages and data sources"

**🔴 Haqiqat:** [github.com/SamuelPfisterer/EuroSpeech](https://github.com/SamuelPfisterer/EuroSpeech):
- **Litsenziya: "currently TBD (To Be Determined)"** — ya'ni yuridik jihatdan foydalanish aniq emas
- **Faqat 4 commit**
- O'rnatish ko'rsatmalari umumiy: "Generally, you will need Python (>=3.8 recommended) and pip"
- O'z audio/transkriptni ulash uchun **konfig formati hujjatlashtirilmagan**
- **SLURM** job skriptlari, ixtiyoriy **Supabase** — HPC klaster nazarda tutilgan
- Windows haqida hech narsa; Linux/Unix naqshlari dominant
- Modullar: `data_sourcing/`, `download_pipeline/`, `alignment_pipeline/`, `asr-fine-tuning/`, `downloading_aligning_per_parliament/`

**Cheklovlar (muallif tan oladi):**
1. **Domen:** "derived entirely from parliamentary recordings, a domain characterized by formal, planned, and often repetitive speech"
2. **Geografik qamrov** Evropa bilan cheklangan; dialekt o'zgaruvchanligi cheklangan
3. **🔴 ASR'ga bog'liqlik:** "The alignment process is **dependent on the quality of existing ASR models**" — past resursli tillarda yoki buzilgan audioda "accuracy of alignments may be reduced, potentially impacting the quality of the resulting training data"
4. Formal diarizatsiyani talab qilmaydi (ixtiyoriy), spontan nutq bilan yaxshi ishlamaydi

### 11.2 WorldSpeech — va O'ZBEK tili

**Maqola:** *WorldSpeech: A Multilingual Speech Corpus from Around the World* — 2026, [arXiv:2605.09167](https://arxiv.org/pdf/2605.09167).

**Bosqichlar:**
1. **Collection & standardization** — mono **24 kHz**; transkriptlar plain text'ga; PDF ustun aniqlash; **OCR** muammoli yozuvlar uchun (Nastaliq urdu, birman)
2. **Segmentation & ASR** — **1–10 soatlik** uzun audio **Silero VAD** bilan **3–30 sekund**ga bo'linadi; har bir segment **tilga xos ASR modeli** bilan transkripsiya qilinadi (model ablation orqali tanlanadi)
3. **Audio-transcript pairing** — API, metadata, **fuzzy string matching**, yoki ketma-ket raqamlash orqali
4. **CER-based matching** — sliding-window CER; segment **CER < 0.3** bo'lsa saqlanadi (inson matni ground truth)
5. **🟢 Iterative alignment refinement** (ixtiyoriy) — boshlang'ich ASR'ni birinchi o'tish natijalarida fine-tune qilib, **bir xil audioni qayta moslashtirish** → avval topilmagan segmentlarni tiklash. Natija: **har bir til uchun 19.5%–201.1% qo'shimcha moslashtirilgan ma'lumot**, qo'shimcha ma'lumot yig'ishsiz. ⚠️ **Zaif ASR'li tillar uchun eng katta foyda** — bu aynan o'zbek tili holati.

**Hajm:** **65,072.4 soat**; **88 alohida til** (76 tasi 10 soatdan yuqori), **127 til variatsiyasi**, **82 davlat**.

**🟢 O'ZBEK (uz_uz):**
| Maydon | Qiymat |
|---|---|
| Soat | **33.7 h** |
| Davlat | O'zbekiston |
| **DNSMOS OVR** | **2.84** |
| Manba | **RFE/RL (Ozodlik)** — "RFE/RL Terms of Use (**non-commercial**)" |

Yuklash: `load_dataset("disco-eth/WorldSpeech", "uz_uz", split="train")`. Litsenziya **CC BY-NC 4.0**. Boshqa turkiy tillar: turk, ozarbayjon, qirg'iz, turkman, qozoq.

⚠️ **33.7 soat va DNSMOS 2.84** — bu Emilia chegarasi (3.0) dan **past**. Ya'ni WorldSpeech'ning o'zbekcha ma'lumoti TTS uchun qo'shimcha tozalashsiz yaroqsiz; u ASR uchun qurilgan.

**🔴 Kod:** [github.com/ETH-DISCO/worldspeech](https://github.com/ETH-DISCO/worldspeech) — faqat `eval/` (Whisper FLEURS/CommonVoice evaluation) va `train/` (SLURM wrapper). **2 commit, 1 star, "Citation: coming soon".** Beshta qadamli pipeline README'da **tasvirlangan**:
> "01_build_csv → 02_download (24 kHz mono WAV) → 03_verify (spot-check, ASR, CER < 0.3) → 04_align (VAD → ASR → sliding-window CER) → 05_upload (segmentlarni kesish, parquet)"

...lekin **CLI/konfig fayllari repoda yo'q**; implementatsiya **EuroSpeech aligner**ga havola qiladi. 5 konfig (xmer, lao, birman, vyetnam, uyg'ur) **faqat metadata** — audioni `source_url` dan o'zingiz yuklaysiz.

**Cheklovlar:** parlament/radio-eshittirish og'ishi (spikerlar katta yoshli, formal ta'limli, ommaviy kasblardan); formal nutq registri; **"Alignment quality is bounded by the off-the-shelf ASR used during alignment, which varies substantially by language"**.

---

## 12. Lhotse va DataSpeech — infratuzilma va annotatsiya qatlamlari

### 12.1 Lhotse — pipeline emas, konteyner

[github.com/lhotse-speech/lhotse](https://github.com/lhotse-speech/lhotse), **Apache-2.0**, `pip install lhotse`, **Python 3.10+**. k2-fsa / Johns Hopkins hamjamiyati (next-generation Kaldi ekotizimi).

**Asosiy abstraksiyalar:**
| Klass | Vazifa |
|---|---|
| `Recording` / `RecordingSet` | audio yozuv metadata: yo'l, sampling rate, kanallar, davomiylik |
| `SupervisionSegment` / `SupervisionSet` | vaqt oralig'i + transkript, spiker ID, til |
| `Cut` / `CutSet` | **"workhorse"** — oynalash va transformatsiyalar orqali moslashuvchan manipulyatsiya |
| `FeatureSet` | diskda saqlangan akustik xususiyatlar |

**Operatsiyalar:** `trim_to_supervisions`, mixing, perturbation, feature extraction, on-the-fly va oldindan hisoblangan augmentatsiya, feature-space cut mixing, odam o'qiy oladigan manifest I/O, **Lhotse Shar / WebDataset** eksporti, dataset blending, samarali bucketing.

**100+ korpus retsepti** (hujjatlarda ~130 korpus ro'yxatlangan): LibriSpeech, CommonVoice, AMI, VoxCeleb, MUSAN, Fisher, Switchboard, TED-LIUM va boshqalar. CLI: `lhotse download <corpus>` va `lhotse prepare <corpus>` → `download_<corpus>()` / `prepare_<corpus>()` funksiyalariga o'raladi; chiqish `{'recordings': RecordingSet, 'supervisions': SupervisionSet}`.

**O'z datasetingiz uchun:** `Recording` va `SupervisionSegment` obyektlarini qo'lda tuzasiz, `RecordingSet`/`SupervisionSet` ga yig'asiz, JSON/JSONL ga serializatsiya qilasiz. ([docs](https://lhotse.readthedocs.io/en/latest/corpus.html))

**Pozitsiya:** 🟢 Lhotse — **to'g'ri ma'lumot modeli**, lekin **sifat filtri, ASR, alignment, enhancement yo'q**. Bu sizning pipeline'ingizning *chiqish formati* bo'lishi kerak, pipeline'ning o'zi emas. Windows qo'llab-quvvatlashi hujjatlashtirilmagan (ma'lumot topilmadi).

### 12.2 DataSpeech — sifat atributlari bilan annotatsiya

[github.com/huggingface/dataspeech](https://github.com/huggingface/dataspeech), **MIT**.

**Hisoblanadigan metrikalar:** gapirish tezligi (fonema/davomiylik), **SNR**, **C50** (reverberatsiya), **pitch** (o'rtacha va std), **PESQ**, **SI-SDR**.

**Uch qadam:**
1. `main.py` — uzluksiz o'zgaruvchilarni hisoblaydi (gapirish tezligi CPU worker'larda; pitch/SNR/reverb GPU'da)
2. `scripts/metadata_to_text.py` — uzluksiz qiymatlarni **histogramma asosida** kategorik bin'larga aylantiradi (masalan gapirish tezligi **7 bin**: "very slowly" → "very fast"); bin chegaralari **dataset statistikasidan** aniqlanadi
3. `scripts/run_prompt_creation.py` — LLM bilan kategorik teglardan tabiiy tildagi tavsif yaratadi

**Qayta ishlatish:** ✅ **har qanday HF `datasets`-mos dataset** (audio + text ustunlari bilan). Sof Python, `requirements.txt`. 🟢 **Windows'da ishlashi ehtimoli eng yuqori** — og'ir sistem bog'liqliklari yo'q.

**Maqsad:** Parler-TTS'ni o'qitish; Lyth & King'ning sintetik annotatsiyalar bo'yicha ishini takrorlash.

⚠️ **Maintenance:** oxirgi hujjatlashtirilgan yangilanish **2024 avgust** (Parler-TTS uchun v1 yaxshilanishlari), 136 commit → **yarim uxlab qolgan**.

⚠️ **Muhim:** DataSpeech **tozalamaydi** — annotatsiya qiladi va filtrlashni **sizga** qoldiradi. ASR yo'q, alignment yo'q, segmentatsiya yo'q, source separation yo'q.

### 12.3 OpenBibleTTS — past resursli tillar uchun Bibliya pipeline'i

**Maqola:** [arXiv:2606.09553](https://arxiv.org/pdf/2606.09553) (2026) — Guzmán, Beyene, Alabi, Jeon, Klakow, Adelani.

**37 til, 5 region, 3,469 soat, 1,121,956 utterance.**

**Bosqichlar:**
1. **Text extraction** — USFM/USX muqaddas yozuv fayllaridan oyat darajasidagi matn
2. **Alignment (ikki parallel usul):**
   - **Timing-file usuli** (28 til): oyat chegaralarini so'z darajasidagi timestamp JSON bilan kesishtirish, MP3'larni **ffmpeg** bilan segmentatsiya
   - **Forced alignment** (9 til): **ReadAlongs Studio** zero-shot aligner, CTC asosidagi dinamik programmalash; maxsus g2p modeli yo'q tillar uchun **universal grafema-fonema fallback (`und`)**
3. **Speaker diarization** — `pyannote/speaker-diarization-precision-2`, har bir Bibliya kitobi uchun alohida
4. **Quality filtering** — "Best" subset uchun **to'rtta mezon, barchasi o'tishi kerak**:
   - davomiylik **≤ 30 sekund**
   - transkript **≥ 10 belgi**
   - **CTC feasibility** tekshiruvi
   - audio/matn nisbati **til bo'yicha o'rtachadan ±3 standart og'ish** ichida

**Kod:** [open-bible-resources](https://github.com/McGill-NLP/open-bible-resources), [open-bible-tts](https://github.com/McGill-NLP/open-bible-tts), [HF](https://huggingface.co/multilingual-tts) — **CC BY-SA 4.0**, derivativ qayta ishlatishga ruxsat.

**Cheklovlar:** "Models trained and evaluated primarily within constrained, scripture-like domain" — tor lug'at, formal stil; yagona ASR metrikasi (`omniASR_LLM_1B_v2`) model-maxsus og'ish kiritishi mumkin; ndebele WER baholashdan chiqarib tashlangan. Saqlanish darajasi — **ma'lumot topilmadi**.

🟢 **O'zbekcha uchun ahamiyati:** agar o'zbekcha Bibliya audio+matn mavjud bo'lsa, bu pipeline'ni to'g'ridan-to'g'ri ishlatish mumkin (ReadAlongs + `und` fallback). Lekin domen juda tor.

---

## 13. Konsensus pipeline — 8+ mustaqil laboratoriya nimada kelishgan

Quyidagi diagramma yuqoridagi barcha pipeline'larda **mustaqil ravishda qayta-qayta paydo bo'lgan** bosqichlardan olingan. Agar 8 ta mustaqil laboratoriya bir xil ketma-ketlikka kelgan bo'lsa — bu to'g'ri dizayn.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  0. MANBA YIG'ISH / MEDIA OLISH                                             │
│     crawl (YODAS) · API (HiFiTTS-2: LibriVox) · qo'lda CSV (EuroSpeech)     │
│     → 🔴 BU BOSQICH HAR DOIM INSON ISHI                                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. STANDARTLASHTIRISH                                                       │
│     mono · sample rate (16k ASR / 24k–44.1k TTS) · bit depth · loudness      │
│     Emilia: 24kHz/-20dBFS · Granary: 16kHz · WorldSpeech: 24kHz              │
│     ✅ barcha 14 pipeline'da bor                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. [IXTIYORIY] MANBA AJRATISH / SHOVQIN TOZALASH / TIKLASH                   │
│     UVR-MDX (Emilia) · MBTFNet (WenetSpeech4TTS)                            │
│     HTDemucs + VoiceFixer + DeepFilterNet3 (IndicVoices-R)                  │
│     Miipher (LibriTTS-R)                                                     │
│     ⚠️ Libriheavy / Granary / HiFiTTS-2 / EuroSpeech / LEMAS BU BOSQICHNI    │
│        TUSHIRIB QOLDIRADI. LEMAS aniq aytadi: "we do not apply explicit      │
│        speech enhancement or background noise removal"                       │
│     → TTS uchun KERAK, ASR uchun KERAK EMAS                                  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. SPIKER DIARIZATSIYA / KO'P-SPIKER ANIQLASH                               │
│     pyannote-3.1 (Emilia) · ResNet293+spektral klaster (WenetSpeech4TTS)     │
│     Sortformer (HiFiTTS-2) · pyannote-precision-2 (OpenBibleTTS)            │
│     🔴 BARCHASI BU BOSQICHDA YETARLI EMAS (§14 ga qarang)                    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. VAD SEGMENTATSIYA → 3–30 SEKUND                                          │
│     🟢 SILERO VAD — DE-FACTO STANDART (Emilia, Granary, EuroSpeech,          │
│        WorldSpeech)                                                          │
│     Emilia 3–30s · EuroSpeech 3–20s · WorldSpeech 3–30s · Libriheavy 2–30s  │
│     Granary maks 40s · HiFiTTS-2: nuqtadan keyin ≥0.08s pauzada             │
│     ✅ 3–30 SEKUND — UNIVERSAL KONSENSUS                                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. ASR / PSEUDO-LABEL                                                       │
│     Whisper medium+WhisperX (Emilia) · FasterWhisper large-v3 (Granary)     │
│     Whisper Turbo (EuroSpeech) · Paraformer-large (WenetSpeech4TTS)         │
│     Parakeet-TDT-CTC (HiFiTTS-2, SDP) · XLS-R (MMS)                         │
│     🔴 ENG KRITIK BOSQICH — BUTUN PIPELINE SIFATI SHUNGA BOG'LIQ            │
│        EuroSpeech: "dependent on the quality of existing ASR models"        │
│        WorldSpeech: "bounded by the off-the-shelf ASR used during alignment" │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
        ┌──────────────────────────┴──────────────────────────┐
        │ Referens matn bormi?                                 │
        └──────┬───────────────────────────────────┬───────────┘
          HA   ▼                                   ▼  YO'Q
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│ 6a. MATN BILAN MOSLASHTIRISH     │   │ 6b. ASR CHIQISHINI LABEL DEB     │
│  (6-bosqich)                     │   │     QABUL QILISH                 │
│                                  │   │   + gallyutsinatsiya filtri      │
│ · suffix array + Levenshtein     │   │     (Granary: takror n-gram,     │
│   (Libriheavy / text_search)     │   │      uzun so'z, qo'lda yozilgan  │
│ · sliding-window CER             │   │      ibora ro'yxatlari)          │
│   (EuroSpeech 2-bosqichli,       │   │   + char-rate / charset filtri   │
│    WorldSpeech CER<0.3)          │   │   + LID tekshirish (lid_prob≥0.8)│
│ · CTC forced alignment           │   │                                  │
│   (MMS uroman+GPU Viterbi,       │   │ ⚠️ YODAS: avtomatik subtitrlar   │
│    NeMo Forced Aligner,          │   │    32.3% CER, deletion 26.6      │
│    ReadAlongs `und`)             │   │                                  │
│ · CTC loss threshold (YODAS 2.0) │   │                                  │
│                                  │   │                                  │
│ 🟢 PUNKTUATSIYA BEPUL KELADI     │   │ 🔴 PUNKTUATSIYANI TIKLASH KERAK  │
│    (Libriheavy, HiFiTTS-2 87%)   │   │                                  │
└──────────────┬───────────────────┘   └──────────────┬───────────────────┘
               └───────────────┬───────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. MATN NORMALIZATSIYA + PUNKTUATSIYA/KATTA HARF TIKLASH                    │
│     NeMo ITN/TN · WFST (GLOBE) · NeMo DistilBERT (HiFiTTS-2, 13%)           │
│     Qwen2.5-7B-Instruct (Granary) — ⚠️ CER farqi >5% bo'lsa ASLIGA QAYTADI  │
│     🔴 TILGA XOS — HAR BIR TIL UCHUN QO'LDA ISH                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  8. SIFAT BALLARINI HISOBLASH (filtrlamay, faqat o'lchash)                    │
│     🟢 DNSMOS P.835 OVRL — DE-FACTO STANDART (Emilia, WenetSpeech4TTS,      │
│        WorldSpeech)                                                          │
│     UTMOS (GLOBE_V3) · SNR · C50 · signal bandwidth (GLOBE, HiFiTTS-2)      │
│     PESQ · SI-SDR · Torch-Squim (SDP) · pitch mean/std · gapirish tezligi    │
│     → DataSpeech bu bosqichni mustaqil kutubxona qilib ajratgan              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  9. FILTRLASH + POG'ONALARGA BO'LISH                                         │
│     Emilia: DNSMOS≥3.0 (yoki 2.4) · WenetSpeech4TTS: ≥4.0/3.8/3.6           │
│     EuroSpeech: CER<10/20/30% · GLOBE: SNR≥0dB, BW≥12kHz, edit dist≤1       │
│     IndicVoices-R: C50≥30dB, SNR≥25dB, CER≤5% · MMS: CER≤10%, ball≥−0.2     │
│     HiFiTTS-2: BW≥11/13kHz, CER<100%                                         │
│     🟢 POG'ONALI RELIZ (tier) — FILTRLASH O'RNIGA ENG YAXSHI AMALIYOT:       │
│        barcha metrikalarni metadata qilib bering, foydalanuvchi o'zi tanlasin │
│        (HiFiTTS-2 va Granary shunday qiladi)                                 │
│     🔴 BARCHA CHEGARALAR QO'LDA TANLANGAN — PRINSIPIAL USUL YO'Q            │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  10. MANIFEST / PAKETLASH                                                    │
│      Lhotse CutSet · NeMo manifest JSONL · parquet (WorldSpeech)             │
│      tar shards (SDP: convert_to_audio_tarred_dataset, num_shards)           │
│      Lhotse Shar / WebDataset                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                        ┌──────────┴──────────┐
                        │  🔴 11. INSON       │
                        │  TINGLASH / SPOT-   │
                        │  CHECK              │
                        │  Hech bir pipeline  │
                        │  buni avtomatlash-  │
                        │  tirmagan (§14)     │
                        └─────────────────────┘
```

### 13.1 Konsensus — bir qatorda

> **standartlashtirish → [tozalash] → diarizatsiya → VAD segmentatsiya (3–30s) → ASR → [matn bilan moslashtirish] → matn normalizatsiya → sifat ballari → chegara/pog'ona → manifest**

### 13.2 Qaysi vositalar de-facto standart bo'lgan

| Bosqich | G'olib | Necha pipeline'da |
|---|---|---|
| VAD | **Silero VAD** | Emilia, Granary, EuroSpeech, WorldSpeech, SDP |
| Diarizatsiya | **pyannote** | Emilia, OpenBibleTTS, SDP |
| ASR | **Whisper variantlari** | Emilia, Granary, EuroSpeech, WorldSpeech, GLOBE |
| Sifat bali | **DNSMOS P.835 OVRL** | Emilia, WenetSpeech4TTS, WorldSpeech |
| Segment uzunligi | **3–30 sekund** | Emilia, WorldSpeech, Libriheavy (2–30), EuroSpeech (3–20) |
| Alignment | **CTC forced alignment** / sliding-window CER | MMS, HiFiTTS-2, OpenBibleTTS, LEMAS / EuroSpeech, WorldSpeech |
| Manifest | **NeMo JSONL** yoki **Lhotse CutSet** | SDP / icefall ekotizimi |

---

## 14. Qaysi bosqichni hammasi hali ham noto'g'ri qiladi yoki insonga qoldiradi

### 14.1 🔴 Ko'p-spikerli / ustma-ust nutq — hamma tan oladi, hech kim hal qilmaydi

- **Emilia** ([arXiv:2501.15907](https://arxiv.org/html/2501.15907v1)): pyannote-3.1 dan keyin ham "a **small proportion** of speech segments containing more than one speaker or overlaps" qoladi
- **WenetSpeech4TTS**: ko'p-spikerli segmentlarni **aniqlaydi va tashlaydi** — tuzatmaydi
- **HiFiTTS-2**: ~**104.4 soat**ni ko'p-spikerli deb aniqlaydi va **faqat metadata sifatida teglaydi** — tashlamaydi ham, tuzatmaydi ham
- **EuroSpeech**: diarizatsiyani umuman **majburiy qilmaydi** (ixtiyoriy)

### 14.2 🔴 Transkript to'g'riligi — proksi metrikalar, ground truth emas

Hech bir pipeline transkriptning **haqiqatan to'g'ri** ekanini bilmaydi. Barchasi proksi ishlatadi:
- ASR ishonch bali (WenetSpeech4TTS: < 0.7 tashlanadi)
- ikki transkript orasidagi CER (IndicVoices-R: > 5%, MMS: > 10%, EuroSpeech: > 30/20/10%)
- edit distance (GLOBE: > 1)

Bu **aylanma mantiq**: ASR'ni ASR bilan tekshirish. **Granary** buni tan oladi va eng ochiq-oydin inson mehnatini ko'rsatadi: **tilga xos qo'lda yozilgan gallyutsinatsiya iboralari ro'yxatlari** (masalan "Sous-titrage Société Radio-Canada"). Bu avtomatlashtirilmagan — har bir til uchun inson yozadi.

### 14.3 🔴 Chegaralarni tanlash — prinsipial usul yo'q

| Pipeline | DNSMOS / CER chegarasi | Qanday tanlangan? |
|---|---|---|
| Emilia (SLT) | DNSMOS ≥ **3.0** | asoslanmagan |
| Emilia-Large | DNSMOS ≥ **2.4** | ⚠️ **o'zgartirilgan, sababi tushuntirilmagan** |
| WenetSpeech4TTS | **4.0 / 3.8 / 3.6** | asoslanmagan |
| EuroSpeech | CER **< 30 / 20 / 10%** | pog'onali reliz orqali qochib ketadi |
| IndicVoices-R | SNR ≥ 25 dB, C50 ≥ 30 dB | asoslanmagan |

Emilia'ning o'zi bir xil pipeline'da chegarani **3.0 → 2.4** ga tushirgan va faqat "the only modification was changing the DNSMOS filtering threshold" deb aytgan. Bu chegaralarning **ilmiy asosi yo'qligini** ko'rsatadi.

### 14.4 🔴 Yakuniy inson tinglashi — hech kim qochib qutula olmagan

- **LibriTTS-R**: "**two of 640** LibriTTS-R speech samples were distorted due to the failure of SR" — **qo'lda tinglash** orqali topilgan, va mualliflar train splitlarida **aniqlanmagan nosozliklar** borligini tan oladi
- **GLOBE_V2**: V1 relizidan keyin "**removed some samples (~5%)** due to the volume and text alignment issues" — ya'ni avtomatik filtr o'tkazib yuborgan, keyin odam topgan
- **WorldSpeech**: pipeline'ning 3-qadami aniq nomi — "**03_verify Spot-check pairs**"

### 14.5 🔴 Manba yig'ish va litsenziya

**EuroSpeech** 22 Evropa parlament saytini **qo'lda tekshirdi**. **WorldSpeech** har bir manba uchun alohida yig'ish skriptlari yozdi. **Omnilingual ASR** umuman avtomatlashtirmadi — **pul to'lab mahalliy hamkorlardan yozdirib oldi**.

### 14.6 🔴 Past resursli tilda ASR — asosiy to'siq

Bu eng muhim qolgan bo'shliq, va **ikkita mustaqil guruh buni aniq yozgan**:
- **EuroSpeech**: "The alignment process is **dependent on the quality of existing ASR models**... accuracy of alignments may be reduced, potentially impacting the quality of the resulting training data"
- **WorldSpeech**: "Alignment quality is **bounded by the off-the-shelf ASR** used during alignment, which **varies substantially by language**"

O'zbek tili uchun bu hal qiluvchi: Whisper o'zbekchada **90.2% WER** ko'rsatadi (loyihaning `TADQIQOT.md` dagi avvalgi topilma). Ya'ni **5-bosqich (ASR) butun konsensus pipeline'ni o'zbekchada ishlashdan to'xtatadi** — qolgan 9 bosqich tayyor bo'lsa ham.

### 14.7 ⚪ Prosodiya / emotsiya / stil belgilash

Faqat **DataSpeech** urinib ko'rgan (pitch, gapirish tezligi → 7 bin → LLM tavsifi), va u ham juda qo'pol. Hech bir dataset pipeline'i emotsiya, aksent-ichki o'zgarish, yoki gapirish stilini ishonchli belgilamaydi.

---

## 15. Nega bular MAHSULOT emas, TADQIQOT ARTEFAKTI?

### 15.1 To'g'ridan-to'g'ri dalillar

| Pipeline | Tadqiqot artefakti ekanining belgisi |
|---|---|
| **Emilia-Pipe** | README'da TODO ro'yxati: "pipeline speed optimization, audio larger than **4GB** support, better noise handling, extended ASR length **beyond 30 seconds**, enhanced labeling" — bu mahsulot roadmap'i emas, tugallanmaganlik tan olinishi. `env.sh` bash skripti, 2 ta onnx checkpointni **qo'lda** yuklash, HF token kerak. |
| **IndicVoices-R** | 🔴 **Pipeline kodi umuman reliz qilinmagan.** Repoda faqat `VoiceCraft/` (model fine-tuning) va `data_links.txt`. Bosqichlar maqolada tasvirlangan, implementatsiya yo'q. |
| **GLOBE** | 🔴 **Pipeline kodi reliz qilinmagan.** "We will release GLOBE publicly after acceptance" — dataset relizi, kod relizi emas. |
| **WenetSpeech4TTS** | 🔴 Pipeline kodi relizi topilmadi; HF kartasida mualliflar "defend and indemnify the authors" talabini qo'yadi — bu huquqiy himoya, mahsulot qo'llab-quvvatlashi emas. |
| **LibriTTS-R** | 🔴 **Miipher kodi/vaznlari relizi maqolada ko'rsatilmagan** — eng qimmatli bosqich takrorlanmaydi. |
| **YODAS** | ⚠️ "data collection software **will be** open to the public" — maqoladagi kelasi zamondagi **va'da**, repo havolasi yo'q. |
| **Libriheavy** | ⚠️ Kod ochiq (Apache-2.0) lekin audio katalogi **`download/librilight` ga hardcoded**; 14 commit. |
| **EuroSpeech** | 🔴 **Litsenziya "currently TBD"**, **4 commit**, o'rnatish ko'rsatmalari yo'q, SLURM skriptlari, konfig formati hujjatlashtirilmagan — hammasi "maqola chiqdi, ish tugadi" signallari. |
| **WorldSpeech** | 🔴 **2 commit, 1 star, "Citation: coming soon"**; pipeline README'da tasvirlangan, lekin CLI yo'q — implementatsiya boshqa repoga (EuroSpeech) havola qiladi. |
| **LEMAS** | ⚠️ Maqolada GitHub/HF havolalari eslatilgan, lekin pipeline uchun **aniq URL berilmagan**. |
| **NeMo SDP** | 🟢 **YAGONA ISTISNO.** Apache-2.0, **221 star, 219 commit**, faol, YAML'da `test_cases` validatsiyasi, hujjatlangan sayt, 14+ dataset konfigi. Sababi aniq: NVIDIA buni **o'zi uchun** ishlatadi (Canary, Parakeet modellarini o'qitish uchun) — ya'ni ichki iste'molchisi bor. |
| **MMS** | 🟢 **Ikkinchi istisno:** GPU forced aligner **TorchAudio ichiga qo'shilgan** — ya'ni `pip install torchaudio` bilan ishlaydi. Bu "tadqiqot → infratuzilma" ga o'tishning yagona to'liq misoli. |

### 15.2 Strukturaviy sabab

Bu maqolalarning **mahsuloti dataset, pipeline emas**. Pipeline — datasetga yetish uchun vosita. Shuning uchun:
1. **Maqola qabul qilingandan keyin rag'bat tugaydi** — EuroSpeech'ning 4 commit'i, WorldSpeech'ning 2 commit'i.
2. **Datasetni relizi qilish kifoya** — IndicVoices-R va GLOBE pipeline'ni umuman bermagan, chunki ularga baho beriladigan artefakt dataset.
3. **Reproducibility talabi faqat natijaga, vositaga emas** — konferensiyalar datasetni so'raydi, pipeline'ning o'rnatiladigan bo'lishini so'ramaydi.
4. **Faqat ichki iste'molchisi bo'lgan tashkilot vositani boqadi** — NVIDIA (SDP), Meta (TorchAudio aligner), k2-fsa (Lhotse).

---

## 16. Windows'dagi yakka ishlab chiquvchi uchun eng qayta ishlatiladigan qaysi?

**Halol baho. Hech biri Windows'da "pip install" bilan to'liq ishlamaydi.**

| O'rin | Vosita | Windows'da o'rnatish qiyinligi | Nimani beradi |
|---|---|---|---|
| 🥇 **1** | **DataSpeech** (MIT) | 🟢 **Oson.** Sof Python + `requirements.txt`, HF `datasets` ustida. Og'ir sistem bog'liqligi yo'q. GPU ixtiyoriy. | 8-bosqich: SNR, C50, pitch, gapirish tezligi, PESQ, SI-SDR + binning. ⚠️ Yarim uxlab qolgan (2024-08). |
| 🥈 **2** | **Lhotse** (Apache-2.0) | 🟢 **Oson.** `pip install lhotse`, Python 3.10+. (Windows rasman hujjatlashtirilmagan, lekin torchaudio ishlaydi.) | 10-bosqich: manifest/CutSet modeli. Butun quvurning **chiqish formati**. |
| 🥉 **3** | **Emilia-Pipe** (Amphion ichida) | 🟡 **O'rta.** `env.sh` bash — qo'lda tarjima qilish kerak. pyannote + Silero + WhisperX + onnxruntime-gpu — barchasi Windows wheel'lari bor. 2 onnx checkpointni qo'lda yuklash + HF token. CUDA kerak. | 1–6 bosqichlarning **to'liq, ishlaydigan referens implementatsiyasi**. `config.json` → `input_folder_path` bilan o'z audiongiz. ⚠️ O'zbek tili `language.supported` da YO'Q. |
| **4** | **`text_search`** (`pip install fasttextsearch`) | 🟡 **O'rta.** PyPI'da bor, lekin Windows wheel mavjudligi hujjatlashtirilmagan; CMake build fallback. | 6a-bosqich: uzun audio ↔ uzun kitob matni moslashtirish (suffix array + Levenshtein). |
| **5** | **MMS GPU aligner (TorchAudio ichida)** | 🟢 **Oson** (torchaudio Windows'da ishlaydi). | 6a-bosqich: `uroman` + CTC forced alignment, 43 daqiqalik audio, O(L) xotira, `⟨*⟩` star token. |
| **6** | **NeMo SDP** (Apache-2.0) | 🔴 **Qiyin.** NeMo'da native Windows o'rnatish ko'rsatmasi **yo'q** — faqat `docker run nvcr.io/nvidia/nemo-speech:...` va `source .venv/bin/activate`. **WSL2 yoki Docker majburiy.** Granary konfigi Qwen2.5-7B + vLLM + CometKiwi talab qiladi. | ✅ **Eng to'liq va umumiy maqsadli**: `tts/ytc` konfigi 10 protsessorli TTS quvuri; Granary konfigi JSON manifest bilan ixtiyoriy audio; **`dataset_configs/uzbek/{fleurs,mcv,uzbekvoice}`** allaqachon bor. |
| **7** | **EuroSpeech aligner** | 🔴 **Juda qiyin.** Litsenziya TBD, o'rnatish hujjati yo'q, SLURM, Supabase, 4 commit. | ✅ **Algoritmik jihatdan eng mos**: ikki bosqichli sliding-window CER alignment — **non-verbatim** uzun matn uchun. Qayta yozish kerak, ko'chirib olish emas. |
| ✖ | **Libriheavy / IndicVoices-R / GLOBE / WenetSpeech4TTS** | ✖ | Hardcoded yoki kod yo'q. Faqat **retsept** sifatida foydali. |

### 16.1 Amaliy tavsiya (Windows, yakka ishlab chiquvchi, o'zbek tili)

1. **Konteyner:** `pip install lhotse` → `Recording` + `SupervisionSegment` + `CutSet`. Butun loyihani shu modelda qurish — keyin icefall/NeMo/Parler-TTS ga bepul ulanadi.
2. **Blueprint:** Emilia-Pipe'ning 6 bosqichini ketma-ketlik sifatida ko'chirish, lekin:
   - 5-bosqich (ASR): Whisper o'rniga **Kotib** (o'zbekcha, loyihadagi avvalgi qarori to'g'ri)
   - 6-bosqichga DNSMOS P.835 OVRL qo'shish (`sig_bak_ovr.onnx` — Emilia repo'sidan)
3. **Alignment (to'liq matn + uzun audio holati):** `ctc-forced-aligner` (loyihada allaqachon tanlangan) + **EuroSpeech'ning ikki bosqichli CER refinement'ini qayta yozish** (±15 so'z start, L±15 oyna) — bu non-verbatim matnga chidamlilik beradi.
4. **🟢 WorldSpeech'ning iterativ qayta moslashtirishini qo'llash:** birinchi o'tishda moslashgan segmentlarda Kotib'ni fine-tune qilib, **bir xil audioni qayta moslashtirish**. Maqola zaif ASR'li tillarda **+19.5% dan +201.1%** gacha qo'shimcha ma'lumot berganini yozadi — o'zbekcha aynan shu holat.
5. **Annotatsiya:** DataSpeech'ni (MIT) o'zgartirmasdan ishlatish → SNR/C50/pitch/rate metadata.
6. **Reliz:** filtrlash **o'rniga** HiFiTTS-2/Granary uslubida **barcha metrikalarni metadata qilib berish + pog'onali subsetlar** (Premium/Standard/Basic yoki CER<10/20/30).
7. **SDP'ga faqat WSL2'da qarash:** `dataset_configs/uzbek/uzbekvoice` konfigini **o'qish** — qanday protsessorlar ishlatilganini bilish uchun, hatto ishga tushirmasa ham.

---

## 17. Yakuniy xulosa

### Nima allaqachon hal qilingan
- Bosqichlar ketma-ketligi — **8+ mustaqil laboratoriya bir xil dizaynga keldi** (§13)
- Har bir bosqich uchun de-facto standart vosita (Silero VAD, pyannote, DNSMOS, CTC forced alignment)
- Chegaralar va retseptlar — nashr etilgan va raqamlangan
- Umumiy maqsadli, konfig-boshqaruvli implementatsiya — **NeMo SDP** (Apache-2.0, faol)
- Manifest/ma'lumot modeli — **Lhotse**
- Sifat annotatsiyasi — **DataSpeech**
- Uzun audio ↔ uzun matn alignment — **MMS (TorchAudio)**, **text_search**, **EuroSpeech algoritmi**

### Haqiqiy qolgan bo'shliq
1. **Past resursli tilda ASR** (5-bosqich) — EuroSpeech va WorldSpeech ikkisi ham pipeline sifati off-the-shelf ASR bilan **cheklanganini** aniq yozgan. O'zbekcha uchun bu **butun masala**.
2. **Tilga xos matn normalizatsiya + punktuatsiya tiklash** (7-bosqich) — Granary 25 EU tili uchun Qwen + qo'lda charset bilan qildi; o'zbekcha uchun hech kim qilmagan (+ yangi alifbo islohoti).
3. **Chegaralarni kalibrlash** — hammasi qo'lda tanlangan, prinsipial usul yo'q.
4. **Windows / bitta GPU / desktop paketlash** — hech kim qilmagan.
5. **Inson tinglashini avtomatlashtirish** — hech kim hal qilmagan (§14.4).

### Nimadan boshlash kerak
**Lhotse** (konteyner) + **Emilia-Pipe bosqich ketma-ketligi** (blueprint) + **Kotib** (5-bosqich, o'zbekcha ASR) + **ctc-forced-aligner** (6a) + **EuroSpeech ikki bosqichli CER refinement** (qayta yozilgan) + **WorldSpeech iterativ refinement** + **DataSpeech** (8-bosqich) + **HiFiTTS-2 uslubidagi pog'onali reliz** (9-bosqich).

---

## Manbalar

**Maqolalar**
- Emilia (SLT 2024) — https://arxiv.org/abs/2407.05361 · HTML: https://arxiv.org/html/2407.05361v1 , https://arxiv.org/html/2407.05361v2
- Emilia kengaytirilgan / Emilia-Large (2025) — https://arxiv.org/abs/2501.15907 · HTML: https://arxiv.org/html/2501.15907v1
- WenetSpeech4TTS (Interspeech 2024) — https://arxiv.org/abs/2406.05763 · HTML: https://arxiv.org/html/2406.05763v2
- Libriheavy (ICASSP 2024) — https://arxiv.org/abs/2309.08105 · HTML: https://arxiv.org/html/2309.08105v2
- IndicVoices-R (NeurIPS 2024) — https://arxiv.org/abs/2409.05356 · HTML: https://arxiv.org/html/2409.05356v1
- GLOBE (Interspeech 2024) — https://arxiv.org/abs/2406.14875 · HTML: https://arxiv.org/html/2406.14875v1 · https://www.isca-archive.org/interspeech_2024/wang24b_interspeech.pdf
- LibriTTS-R (Interspeech 2023) — https://arxiv.org/abs/2305.18802 · HTML: https://ar5iv.labs.arxiv.org/html/2305.18802
- YODAS (ASRU 2023) — https://arxiv.org/abs/2406.00899 · HTML: https://arxiv.org/html/2406.00899v1
- Granary (Interspeech 2025) — https://arxiv.org/abs/2505.13404 · HTML: https://arxiv.org/html/2505.13404v2 · https://www.isca-archive.org/interspeech_2025/raokoluguri25_interspeech.pdf
- HiFiTTS-2 (Interspeech 2025) — https://arxiv.org/abs/2506.04152 · HTML: https://arxiv.org/html/2506.04152v1
- MMS (Meta 2023) — https://arxiv.org/abs/2305.13516 · HTML: https://ar5iv.labs.arxiv.org/html/2305.13516
- Omnilingual ASR (Meta 2025) — https://arxiv.org/pdf/2511.09690
- EuroSpeech (NeurIPS 2025 D&B) — https://arxiv.org/abs/2510.00514 · HTML: https://arxiv.org/html/2510.00514v1 · https://papers.neurips.cc/paper_files/paper/2025/file/58ea63de01321ee52d06b48026981c40-Paper-Datasets_and_Benchmarks_Track.pdf
- WorldSpeech (2026) — https://arxiv.org/pdf/2605.09167 · HTML: https://arxiv.org/html/2605.09167v1
- LEMAS (2026) — https://arxiv.org/abs/2601.04233 · HTML: https://arxiv.org/html/2601.04233v1
- OpenBibleTTS (2026) — https://arxiv.org/pdf/2606.09553 · HTML: https://arxiv.org/html/2606.09553v1

**Kod**
- Emilia-Pipe — https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/README.md
- NeMo SDP — https://github.com/NVIDIA/NeMo-speech-data-processor · https://nvidia.github.io/NeMo-speech-data-processor/
- SDP dataset konfiglari — https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/dataset_configs
- SDP o'zbek konfiglari — https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/dataset_configs/uzbek
- SDP TTS/YTC konfigi — https://raw.githubusercontent.com/NVIDIA/NeMo-speech-data-processor/main/dataset_configs/tts/ytc/config.yaml
- SDP Granary konfigi — https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/dataset_configs/multilingual/granary
- NeMo Speech — https://github.com/NVIDIA-NeMo/NeMo
- Libriheavy — https://github.com/k2-fsa/libriheavy
- text_search — https://github.com/k2-fsa/text_search
- IndicVoices-R — https://github.com/AI4Bharat/IndicVoices-R
- Lhotse — https://github.com/lhotse-speech/lhotse · https://lhotse.readthedocs.io/en/latest/corpus.html
- DataSpeech — https://github.com/huggingface/dataspeech
- EuroSpeech — https://github.com/SamuelPfisterer/EuroSpeech
- WorldSpeech — https://github.com/ETH-DISCO/worldspeech
- MMS — https://github.com/pytorch/fairseq/tree/master/examples/mms
- Omnilingual ASR — https://github.com/facebookresearch/omnilingual-asr
- OpenBibleTTS — https://github.com/McGill-NLP/open-bible-tts · https://github.com/McGill-NLP/open-bible-resources

**Datasetlar (HF / OpenSLR)**
- amphion/Emilia-Dataset — https://huggingface.co/datasets/amphion/Emilia-Dataset
- Wenetspeech4TTS — https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS
- MushanW/GLOBE — https://huggingface.co/datasets/MushanW/GLOBE
- MushanW/GLOBE_V2 — https://huggingface.co/datasets/MushanW/GLOBE_V2
- MushanW/GLOBE_V3 — https://huggingface.co/datasets/MushanW/GLOBE_V3
- LibriTTS-R — http://www.openslr.org/141/
- nvidia/Granary — https://huggingface.co/datasets/nvidia/Granary
- espnet/yodas-granary — https://huggingface.co/datasets/espnet/yodas-granary
- nvidia/hifitts-2 — https://huggingface.co/datasets/nvidia/hifitts-2
- disco-eth/WorldSpeech — https://huggingface.co/datasets/disco-eth/WorldSpeech
- facebook/omnilingual-asr-corpus — https://huggingface.co/datasets/facebook/omnilingual-asr-corpus

**"ma'lumot topilmadi" deb belgilangan joylar**
- WenetSpeech4TTS: pipeline kodi relizi; xom manbadan saqlanish darajasi; inson nazorati bosqichi
- Emilia-Large (216k h): DNSMOS 2.4 chegarasidagi saqlanish darajasi; to'liq 101k/216k uchun xom kirish soatlari
- IndicVoices-R: saqlanish darajasi
- YODAS: saqlanish darajasi; crawler kodining aniq repo havolasi
- LibriTTS-R: Miipher kodi/vaznlari relizi
- LEMAS: saqlanish darajasi; pipeline kodining aniq URL'i
- OpenBibleTTS: saqlanish darajasi
- Omnilingual ASR Corpus: jami soat; sifat nazorati qadamlari
- Lhotse, text_search, Libriheavy, EuroSpeech, WorldSpeech: Windows qo'llab-quvvatlashi rasman hujjatlashtirilmagan

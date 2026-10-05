# Tadqiqot 3: Audio preprocessing va avtomatik SIFAT FILTRLASH / VALIDATSIYA (aniq raqamli thresholdlar bilan)

> Qamrov: segmentatsiya/VAD, reference-siz sifat metrikalari, SNR, ASR back-check (CER/WER), alignment confidence, speaker verification, loudness, sample rate, denoise/enhancement, silence trim.
> Qamrovdan tashqarida (boshqa agentlar): forced alignment tool'lari taqqoslash, mavjud korpuslar tarixi, o'zbek tili xususiyatlari.
>
> **Qoida:** har bir raqam manba URL bilan. Manba topilmagan joyda `ma'lumot topilmadi` deb yozilgan — taxminiy raqam yo'q.

---

## 0. MASTER JADVAL — metric | tool | qabul qilish thresholdi | manba

| # | Metric / tekshiruv | Tool | Qabul qilish thresholdi (accept) | Manba URL |
|---|---|---|---|---|
| 1 | VAD speech ehtimoli | silero-vad | `threshold=0.5` (default) | https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py |
| 2 | VAD threshold rejimlari | NeMo Curator (silero) | `0.3` high-recall / `0.5` balanced / `0.7` strict | https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-filtering/vad |
| 3 | Minimal nutq bo'lagi | silero-vad | `min_speech_duration_ms=250` | yuqoridagi utils_vad.py |
| 4 | Bo'lakni yopish uchun sukunat | silero-vad | `min_silence_duration_ms=100` (default) | yuqoridagi utils_vad.py |
| 5 | Chegara padding | silero-vad | `speech_pad_ms=30` (default) | yuqoridagi utils_vad.py |
| 6 | Chegara padding (TTS uchun) | NeMo Curator | `speech_pad_ms=300` default, **TTS uchun 400–500 ms** | https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-filtering/vad |
| 7 | Segmentlar orasidagi min sukunat | NeMo Curator | `min_interval_ms=500` | yuqoridagi VAD doc |
| 8 | Segment uzunligi | NeMo Curator | `min_duration_sec=2.0`, `max_duration_sec=60.0` (long-form uchun `30.0`) | yuqoridagi VAD doc |
| 9 | Segment uzunligi | Emilia-Pipe | `MIN_SEGMENT_LENGTH=3` s, `MAX_SEGMENT_LENGTH=30` s, `MERGE_GAP=2` s | https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/main.py |
| 10 | Segment uzunligi | EuroSpeech | 3–20 s | https://arxiv.org/html/2510.00514v1 |
| 11 | Segment uzunligi | Transcribe-Align-Segment | > 1 s va < 20 s | https://arxiv.org/html/2406.12674v1 |
| 12 | RMS sukunat chegarasi | audio-slicer (openvpi) | `threshold=-40 dB`, `min_length=5000 ms`, `min_interval=300 ms`, `hop_size=20 ms` (CLI: 10), `max_sil_kept=5000 ms` (CLI: 500) | https://github.com/openvpi/audio-slicer/blob/main/slicer2.py |
| 13 | Energiya chegarasi | auditok `split()` | `energy_threshold=50` dB, `min_dur=0.2` s, `max_dur=5` s, `max_silence=0.3` s | https://github.com/amsehili/auditok |
| 14 | Agressivlik | py-webrtcvad | `mode 0..3` (0 = eng yumshoq, 3 = eng qattiq); frame 10/20/30 ms; sr 8/16/32/48 kHz; 16-bit mono PCM | https://github.com/wiseman/py-webrtcvad |
| 15 | VAD onset/offset | pyannote/segmentation | default `onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0`; tuned: AMI `0.684/0.577`, DIHARD3 `0.767/0.377`, VoxConverse `0.767/0.713` | https://huggingface.co/pyannote/segmentation |
| 16 | Ichki sukunat limiti | GLOBE | uzluksiz ichki sukunat **> 930 ms** bo'lsa utterance o'chiriladi | https://arxiv.org/html/2406.14875v1 |
| 17 | **DNSMOS P.835 OVRL** | Emilia-Pipe | **> 3.0** | https://arxiv.org/html/2407.05361v2 |
| 18 | **DNSMOS** | WenetSpeech4TTS | **≥ 4.0** Premium (945 h) / **≥ 3.8** Standard (4056 h) / **≥ 3.6** Basic (7226 h) / < 3.6 Rest (5574 h) | https://arxiv.org/html/2406.05763v3 |
| 19 | **DNSMOS** | in-the-wild pipeline eval | sinovdan o'tgan to'plam `{2.7, 3.0, 3.2, 3.4}`; **eng yaxshi natija = 2.7** (permissive) | https://arxiv.org/html/2510.03111 |
| 20 | **NISQA** | in-the-wild pipeline eval | sinovdan o'tgan to'plam `{3.0, 3.5, 3.8, 4.2}` | https://arxiv.org/html/2510.03111 |
| 21 | **NISQA** | Crowdsourced-audio-for-TTS | sinov `{2.0, 4.0, 4.4, 4.6}`; eng yaxshi 100 h subset **NISQA ≥ 4.0** | https://arxiv.org/html/2410.13357v1 |
| 22 | **UTMOS** | NeMo Curator | **TTS / voice cloning: `mos_threshold=4.0`**; ASR: `3.5`; web-scraped permissive: `3.0`; umumiy start: `3.5` | https://docs.nvidia.com/nemo/curator/main/curate-audio/process-data/quality-filtering/utmos |
| 23 | SIGMOS | NeMo Curator | misol: `4.0` (noise o'lchovi), `3.5` (overall) | https://docs.nvidia.com/nemo/curator/main/curate-audio/process-data/quality-filtering/ |
| 24 | **WADA-SNR** | LibriTTS | **< 20 dB → "clean" subsetdan chiqariladi**; **< 0 dB → "other" subsetdan ham chiqariladi** | https://ar5iv.labs.arxiv.org/html/1904.02882 |
| 25 | SNR | GLOBE | **SNR < 0 dB → o'chiriladi**; signal bandwidth **< 12 kHz → o'chiriladi** | https://arxiv.org/html/2406.14875v1 |
| 26 | SNR + C50 | IndicVoices-R | **SNR ≥ 25 dB**, **C50 ≥ 30 dB** (reverb) | https://arxiv.org/html/2409.05356 |
| 27 | Pitch / tempo filtrlari | IndicVoices-R | `utterance-pitch-mean ≤ 350 Hz`, `pitch-std ≤ 150 Hz`, `speaking-rate ≤ 30 char/s`, `0.2 < duration < 30 s` | https://arxiv.org/html/2409.05356 |
| 28 | Kompozit audio-sifat bali | ParsVoice | **bal < 75 → TTS uchun ishlatilmaydi**; clipping **> 5% → −20**; silence ratio **> 50% → −15**, `30–50% → −10`; davomiylik `3–15 s → +15`, `15–30 s → +5` | https://arxiv.org/html/2510.10774 |
| 29 | **ASR back-check CER** | Transcribe-Align-Segment | **CER > 30% VA WER > 75% → tashlab yuboriladi**; chegaralarda (ilk/oxirgi 5 belgi) **edge CER > 60% → tashlanadi** | https://arxiv.org/html/2406.12674v1 |
| 30 | **ASR back-check CER** | EuroSpeech | `CER < 30%` → 78.2% data; **`CER < 20%` → asosiy nashr (65.4%)**; `CER < 10%` → 41.0% | https://arxiv.org/html/2510.00514v1 |
| 31 | **ASR back-check WER** | FASA | **σa = 0.1** (WER < 10% → avtomatik qabul), **σi = 0.3** (10–30% → qo'lda tekshirish), > 30% → rad; uzunlik farqi ≤ 1 so'z | https://arxiv.org/html/2406.17926v1 |
| 32 | Matn moslik | GLOBE | Whisper qayta transkripsiya, **edit distance > 1 → o'chiriladi** | https://arxiv.org/html/2406.14875v1 |
| 33 | Matn moslik | LibriTTS | **edit distance = 0** (barcha so'z aynan mos) bo'lsa "aligned" | https://ar5iv.labs.arxiv.org/html/1904.02882 |
| 34 | Ikki transkripsiya orasidagi CER | IndicVoices-R | **CER > 5% → o'chiriladi** | https://arxiv.org/html/2409.05356 |
| 35 | WER filtri | NeMo Curator | `WER ≤ 30%` = high quality; `WER ≥ 80%` → o'chir; pipeline misolida `WER ≤ 75%`; duration `1–30 s` / `≤ 20 s` | https://docs.nvidia.com/nemo/curator/main/curate-audio/process-data/quality-assessment/ |
| 36 | ASR confidence | WenetSpeech4TTS | **confidence < 0.7 → chiqariladi** | https://arxiv.org/html/2406.05763v3 |
| 37 | **CTC-segmentation confidence** | NeMo `run_segmentation.sh` | **`MIN_SCORE=-2`** (undan yuqori score saqlanadi) | https://github.com/NVIDIA/NeMo/blob/main/tools/ctc_segmentation/run_segmentation.sh |
| 38 | **CTC-segmentation confidence** | NeMo `cut_audio.py` | `--threshold` default **`-5`** ("Minimum score value accepted") | https://github.com/NVIDIA/NeMo/blob/main/tools/ctc_segmentation/scripts/cut_audio.py |
| 39 | CTC-seg amalda | Transcribe-Align-Segment | **alignment confidence score > −2** | https://arxiv.org/html/2406.12674v1 |
| 40 | MFA alignment score | Montreal Forced Aligner | **ma'lumot topilmadi** — rasmiy hujjat aniq raqam bermaydi | https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/implementations/alignment_analysis.html |
| 41 | Speaker verification cosine | SpeechBrain `SpeakerRecognition` | **default `threshold=0.25`** (cosine similarity) | https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/inference/speaker.py |
| 42 | Speaker cosine (segment merge) | WenetSpeech4TTS | **> 0.65** → bir xil so'zlovchi deb birlashtiriladi | https://arxiv.org/html/2406.05763v3 |
| 43 | Diarizatsiya klaster merge | WenetSpeech4TTS | klaster markazlari orasidagi cosine **> 0.75** → merge; har klasterdagi eng chetdagi **10%** embedding belgilanadi | https://arxiv.org/html/2406.05763v3 |
| 44 | Speaker cosine (ECAPA-TDNN) | ParsVoice | global merge **0.90**; segment ishonch **c_i < 0.4 → chiqariladi**; cross-book merge uchun **≥ 100 segment** | https://arxiv.org/html/2510.10774 |
| 45 | Intra-speaker cosine (o'lchov) | NeMo TitaNet / ECAPA | TitaNet o'rtacha **0.702**, ECAPA-TDNN **0.658** | https://arxiv.org/pdf/2506.20190 |
| 46 | Loudness target | ffmpeg `loudnorm` | default **`I=-24` LUFS, `TP=-2.0` dBTP, `LRA=7`** | https://ayosec.github.io/ffmpeg-filters-docs/7.1/Filters/Audio/loudnorm.html |
| 47 | Loudness target | TTS-DATA-PIPELINE | qabul qilingan kliplar **−23 LUFS** ga normalizatsiya; rad etilganlar normalizatsiya qilinmaydi | https://github.com/harika1903/TTS-DATA-PIPELINE |
| 48 | Amplituda target | Emilia-Pipe | **`target_dBFS = -20`**, gain **±3 dB** bilan cheklangan, 16-bit, 24 kHz | https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/main.py |
| 49 | Loudness (TTS output, Google) | Google Assistant | stereo **−16 LUFS**, mono **−19 LUFS** | https://developers.google.com/assistant/tools/audio-loudness |
| 50 | Loudness meter | pyloudnorm | ITU-R BS.1770-4, gating block default **400 ms** | https://github.com/csteinmetz1/pyloudnorm |
| 51 | DC offset / polarity | LibriTTS | "Normalize the polarity of audio by flipping up-side-down waveforms by ensuring DC offsets are positive" | https://ar5iv.labs.arxiv.org/html/1904.02882 |
| 52 | Clipping detektsiya | Neyshekar korpusi | "≥ **3 ketma-ket full-scale** sample" = flattened run; qattiq mezon: samplelarning **> 0.1%** rail'da | https://arxiv.org/html/2609.14542v1 |
| 53 | Sample rate | Coqui TTS | "It is better to reduce the sample-rate of your dataset to around **16000–22050**" | https://docs.coqui.ai/en/latest/what_makes_a_good_dataset.html |
| 54 | Sample rate (korpus standarti) | LibriTTS / LibriTTS-R / Emilia / WenetSpeech4TTS / GLOBE | **24 kHz** | https://ar5iv.labs.arxiv.org/html/1904.02882 , https://arxiv.org/html/2407.05361v2 , https://arxiv.org/html/2406.05763v3 |
| 55 | Resample sifati | librosa | `res_type` default **`soxr_hq`**; eng yuqori `soxr_vhq`; `soxr_qq`, `polyphase`, `linear`, `zero_order_hold` aliasing berishi mumkin | https://librosa.org/doc/latest/generated/librosa.resample.html |
| 56 | Metrik modellarning sr talabi | DNSMOS / UTMOS / SQUIM | **16 kHz** (SQUIM: "only supports 16000 Hz") | https://pypi.org/project/speechmos/ , https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html |
| 57 | VAD sr talabi | silero-vad | faqat **8000 Hz va 16000 Hz** | https://github.com/snakers4/silero-vad |
| 58 | Enhancement sr | DeepFilterNet | faqat **48 kHz** wav | https://github.com/Rikorose/DeepFilterNet |
| 59 | Ovoz saqlanishi tekshiruvi | in-the-wild pipeline eval | **MCD ≤ 5 dB** (denoise original ovozni buzmaganini tasdiqlash) | https://arxiv.org/html/2510.03111 |
| 60 | Trailing silence | StyleTTS2 finetuning notes | barcha fayl oxiriga **100 ms sukunat** + stop-token qo'shish; `$` token = **5000 sample ≈ 208 ms @ 24 kHz** | https://github.com/yl4579/StyleTTS2/discussions/81 |
| 61 | Chegara optimallashtirish | ParsVoice | chegaradan **max 3 s** kesish, **0.1 s** qadamda qidirish, **max 5 s** kengaytirish | https://arxiv.org/html/2510.10774 |
| 62 | Chegara kengaytirish | WenetSpeech4TTS | segment boshini orqaga surish **max 0.5 s**; gap **< 0.55 s** bo'lsa birlashtirish; birlashtirish **20 s** gacha | https://arxiv.org/html/2406.05763v3 |
| 63 | Tozalangan ma'lumot TTS'ga ta'siri | LibriTTS-R (Miipher) | TTS MOS: LibriTTS **2.59–3.35** → LibriTTS-R **4.06–4.31** (6 so'zlovchi) | https://ar5iv.labs.arxiv.org/html/2305.18802 |

---

## 1. Segmentatsiya / VAD — amalda ishlatiladigan parametrlar

### 1.1 silero-vad (eng ko'p ishlatiladigan)

`get_speech_timestamps()` haqiqiy default'lari (manba: https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py):

```python
threshold: float = 0.5
sampling_rate: int = 16000
min_speech_duration_ms: int = 250
max_speech_duration_s: float = float('inf')
min_silence_duration_ms: int = 100
speech_pad_ms: int = 30
neg_threshold: float = None
window_size_samples: int = 512        # !!! DEPRECATED, DOES NOTHING !!!
min_silence_at_max_speech: int = 98
use_max_poss_sil_at_max_speech: bool = True
```

Rasmiy izohlar (aynan repo'dan):
- `threshold`: "probabilities ABOVE this value are considered as SPEECH"
- `min_speech_duration_ms`: "Final speech chunks shorter min_speech_duration_ms are thrown out"
- `max_speech_duration_s`: "Chunks longer than max_speech_duration_s will be split at the timestamp of the **last silence that lasts more than 100ms** (if any), to prevent aggressive cutting"
- `min_silence_duration_ms`: "In the end of each speech chunk wait for min_silence_duration_ms before separating it"
- `speech_pad_ms`: "Final speech chunks are **padded by speech_pad_ms each side**"
- `neg_threshold`: "If model's current state is SPEECH, values BELOW this value are considered as NON-SPEECH" (hysteresis — chayqalishni kamaytiradi)

**Muhim:** silero-vad faqat **8000 Hz va 16000 Hz** qo'llaydi (https://github.com/snakers4/silero-vad). Ya'ni 48 kHz master fayl'ni VAD uchun 16 kHz ga tushirib, timestamp'larni master rate'ga ko'paytirish kerak.

### 1.2 TTS dataset kesish uchun tavsiya etilgan (default'lardan farqli) qiymatlar

NVIDIA NeMo Curator VAD hujjati (https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-filtering/vad) silero asosidagi pipeline uchun:

| Parametr | Qiymat | Izoh |
|---|---|---|
| `threshold` | `0.3` / `0.5` / `0.7` | high recall (shovqinli/jim audio) / balanced / strict high-precision |
| `min_duration_sec` | `2.0` | shundan qisqa segmentlar tashlanadi |
| `max_duration_sec` | `60.0` (long-form uchun `30.0`) | uzunlari bo'linadi |
| `speech_pad_ms` | `300` default | **TTS uchun 400–500 ms ga oshirish tavsiya etiladi** ("to preserve prosody and natural breathing patterns") |
| `min_interval_ms` | `500` | segmentlar orasidagi minimal sukunat |

Bu topshiriqda so'ralgan **0.2–0.3 s padding** aynan shu tavsiya doirasida: `speech_pad_ms=200..300` xavfsiz, `300` NeMo Curator default'i, TTS uchun `400–500` ham asoslangan.

### 1.3 So'z boshini kesib yubormaslik (word-onset clipping) uchun amaliy retsept

Manbalardan chiqadigan mexanizm (har biri URL bilan):

1. **Hysteresis ishlatish:** `neg_threshold` ni `threshold` dan past qo'yish (silero-vad rasmiy parametri) — jarangsiz portlovchi undoshlar (p/t/k) boshida VAD ehtimoli past bo'lsa ham SPEECH holatidan chiqmaydi. Manba: utils_vad.py docstring.
2. **`min_silence_duration_ms` ni oshirish** (100 → 250–300): silero o'zi "wait for min_silence_duration_ms before separating" deydi, ya'ni bu qiymat qanchalik katta bo'lsa, so'z ichidagi qisqa pauzalarda kesish kamayadi.
3. **Simmetrik padding:** `speech_pad_ms` har ikki tomonga qo'shiladi (rasmiy docstring: "padded by speech_pad_ms each side") — 200–300 ms.
4. **Chegarani kengaytirish (extension), kesish emas:** WenetSpeech4TTS segment boshini orqaga **max 0.5 s** suradi (https://arxiv.org/html/2406.05763v3); ParsVoice chegarani **0.1 s** qadamda **max 5 s** gacha kengaytirib optimal nuqta qidiradi (https://arxiv.org/html/2510.10774).
5. **Chegara CER tekshiruvi:** ilk/oxirgi 5 belgi uchun alohida, yumshoqroq `edge cer_threshold=60` — bu aynan "chegarada alignment noaniqligi" muammosini hal qilish uchun kiritilgan (https://arxiv.org/html/2406.12674v1).

### 1.4 Boshqa VAD/slicer tool'lari

| Tool | Parametrlar (default) | Manba |
|---|---|---|
| **py-webrtcvad** | `mode ∈ {0,1,2,3}` (0 = least aggressive, 3 = most aggressive); sr ∈ {8000, 16000, 32000, 48000}; frame ∈ {10, 20, 30} ms; 16-bit mono PCM majburiy | https://github.com/wiseman/py-webrtcvad |
| **pyannote/segmentation VAD** | `onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0`; dataset'ga tune qilingan: AMI `0.684/0.577`, DIHARD3 `0.767/0.377`, VoxConverse `0.767/0.713` | https://huggingface.co/pyannote/segmentation |
| **auditok `split()`** | `min_dur=0.2` s, `max_dur=5` s, `max_silence=0.3` s, `energy_threshold=50` dB; `max_leading_silence`, `max_trailing_silence` ham bor | https://github.com/amsehili/auditok |
| **audio-slicer (openvpi)** | class: `threshold=-40.0` dB, `min_length=5000` ms, `min_interval=300` ms, `hop_size=20` ms, `max_sil_kept=5000` ms. **CLI default'lari boshqa:** `hop_size=10`, `max_sil_kept=500` | https://github.com/openvpi/audio-slicer/blob/main/slicer2.py |

> **Ziddiyat qaydi:** audio-slicer'da class default (`max_sil_kept=5000`) va CLI default (`500`) bir-biriga mos kelmaydi — repo kodida shunday. TTS uchun `500` ni ishlatish mantiqiy (5 sekund sukunat saqlash TTS uchun zararli).

### 1.5 Korpus pipeline'laridagi segment uzunligi qoidalari

| Korpus | Qoida | Manba |
|---|---|---|
| Emilia-Pipe | `MERGE_GAP=2` s, `MIN_SEGMENT_LENGTH=3` s, `MAX_SEGMENT_LENGTH=30` s | https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/main.py |
| WenetSpeech4TTS | gap `< 0.55` s → merge; merge `20` s gacha; boshni orqaga `max 0.5` s | https://arxiv.org/html/2406.05763v3 |
| EuroSpeech | `3–20` s | https://arxiv.org/html/2510.00514v1 |
| Transcribe-Align-Segment | `> 1` s va `< 20` s | https://arxiv.org/html/2406.12674v1 |
| IndicVoices-R | `0.2 < duration < 30` s | https://arxiv.org/html/2409.05356 |
| ParsVoice (ball tizimi) | `3–15` s → +15 ball, `15–30` s → +5 ball | https://arxiv.org/html/2510.10774 |
| StyleTTS2 finetune | min `1` s dan uzun; optimal `6–10` s (`max_len=800` ≈ 10 s) | https://github.com/yl4579/StyleTTS2/discussions/81 |
| GLOBE | ichki uzluksiz sukunat `> 930` ms → utterance o'chiriladi | https://arxiv.org/html/2406.14875v1 |

---

## 2. Reference-siz (no-reference) obyektiv sifat metrikalari

### 2.1 DNSMOS P.835 — SIG / BAK / OVRL

- Chiqishlar: "i) speech quality (**SIG**), ii) background noise quality (**BAK**), and iii) the overall quality (**OVRL**)". Inson baholari bilan korrelyatsiya: **PCC = 0.94 (SIG)**, **PCC = 0.98 (BAK va OVRL)**. Manba: https://arxiv.org/abs/2110.01763
- Amalda `sig_bak_ovrl.onnx` model fayli + `dnsmos_local.py` skripti ishlatiladi. Manba: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
- Eng oson yo'l (Windows'da muammosiz): `pip install speechmos` → `dnsmos.run(audio, sr=16000)`; qaytaradi: `ovrl_mos`, `sig_mos`, `bak_mos`, `p808_mos`. **16 kHz majburiy.** Manba: https://pypi.org/project/speechmos/
- Shkala: MOS 1–5 (P.835 5 ballik). DNSMOS repo README'da shkala aniq yozilmagan — 1–5 shkalasi UTMOS hujjatidan tasdiqlanadi (https://docs.nvidia.com/nemo/curator/main/curate-audio/process-data/quality-filtering/utmos).

**DNSMOS kesim qiymatlari — manbalar ZIDDIYATDA:**

| Manba | DNSMOS threshold | Kontekst | URL |
|---|---|---|---|
| Emilia-Pipe / Emilia (101k soat) | **OVRL > 3.0** | katta ko'lamli in-the-wild TTS korpus | https://arxiv.org/html/2407.05361v2 |
| WenetSpeech4TTS | **≥ 4.0 / ≥ 3.8 / ≥ 3.6** (Premium/Standard/Basic) | bir korpusni sifat bo'yicha 4 bosqichga ajratish | https://arxiv.org/html/2406.05763v3 |
| "Evaluation of preprocessing pipelines in the creation of in-the-wild TTS datasets" | **2.7** eng yaxshi ishladi | to'g'ridan-to'g'ri eksperimental taqqoslash | https://arxiv.org/html/2510.03111 |

> **Ziddiyat sharhi:** WenetSpeech4TTS eng qattiq (≥4.0), Emilia o'rtacha (3.0), in-the-wild eval esa eng yumshoq (2.7) ni tavsiya qiladi. Oxirgi maqola sababini ham aytadi: "small adjustments to a DNSMOS threshold can produce larger relative changes in the set of accepted utterances than equivalent adjustments to a NISQA threshold" va "**denoising variants with permissive filtering provide the best overall compromise**" — ya'ni qattiq filtr dataset hajmini shunchalik kamaytiradiki, sifat o'sishi buni qoplamaydi. Xulosa: **avval denoise qilib, keyin yumshoq filtrlash** (DNSMOS ≥ 2.7–3.0) — bitta so'zlovchili, cheklangan hajmdagi dataset uchun to'g'ri strategiya.

### 2.2 UTMOS

- Shkala **1.0–5.0**; NeMo Curator rasmiy sifat xaritasi (https://docs.nvidia.com/nemo/curator/main/curate-audio/process-data/quality-filtering/utmos):

| MOS oralig'i | Sifat |
|---|---|
| 4.0–5.0 | Excellent |
| 3.5–4.0 | Good |
| 3.0–3.5 | Acceptable |
| 2.0–3.0 | Poor |
| < 2.0 | Bad |

- Tavsiya etilgan `mos_threshold`:
  - **Voice cloning / TTS (strict): `4.0`** — "training data sensitive to background noise and clipping"
  - General ASR: `3.5` — "Default works well"
  - Web-scraped (permissive): `3.0` + downstream SIGMOS filtri
  - Umumiy boshlang'ich nuqta: **`3.5`** — "drops obviously distorted, noisy, or clipped audio while keeping most usable training material"
- Checkpoint nomi: `utmos22_strong`; `sample_rate=16000` ("Do not change unless you have a custom checkpoint").
- Asl repo: https://github.com/sarulab-speech/UTMOS22 — VoiceMOS Challenge 2022'da "1st places in 10/16 metrics".

### 2.3 NISQA

- `nisqa.tar` — Overall Quality + **Noisiness, Coloration, Discontinuity, Loudness** (4 o'lcham); `nisqa_mos_only.tar` — faqat overall (finetune uchun); `nisqa_tts.tar` — sintetik nutq naturalness'i. Manba: https://github.com/gabrielmittag/NISQA
- Kesim qiymatlari:
  - in-the-wild eval sinovdan o'tkazgan: **{3.0, 3.5, 3.8, 4.2}** (https://arxiv.org/html/2510.03111)
  - Crowdsourced-audio-for-TTS sinovdan o'tkazgan: **{2.0, 4.0, 4.4, 4.6}**; eng yaxshi 100 soatlik subset **NISQA ≥ 4.0** da olindi (UTMOS 2.94 bilan g'olib) (https://arxiv.org/html/2410.13357v1)
- NISQA vs DNSMOS xatti-harakati: DNSMOS o'zgarishi dataset hajmiga NISQA'dan kuchliroq ta'sir qiladi (yuqoridagi in-the-wild maqola).

### 2.4 SQUIM (torchaudio)

- `SQUIM_OBJECTIVE` → **STOI (0–1)**, **PESQ (~1–4.5)**, **SI-SDR (dB)**. `SQUIM_SUBJECTIVE` → **MOS (1–5)**, "Non-Matching References" usuli bilan (reference audio kerak, lekin bir xil kontent bo'lishi shart emas).
- Misol qiymatlar (tutorialdan): 20 dB SNR da STOI `0.961`, PESQ `2.780`, SI-SDR `20.69` dB, MOS `4.31`; −5 dB SNR da MOS `3.29`.
- **"Currently, Torchaudio-Squim model only supports 16000 Hz sampling rate."**
- Manba: https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html
- SQUIM uchun rasmiy accept/reject threshold: **ma'lumot topilmadi**. IndicVoices-R PESQ/SI-SDR/SNR ni "aggregate" qilib ishlatadi, lekin bitta kesim raqami bermaydi.

### 2.5 WV-MOS

- wav2vec2.0 fine-tune qilingan MOS prediktor. O'rnatish: `pip install git+https://github.com/AndreevP/wvmos`; API: `model.calculate_one(path)`, `model.calculate_dir(dir, mean=True)`. Manba: https://github.com/AndreevP/wvmos
- Shkala va accept threshold: **ma'lumot topilmadi** (README'da yozilmagan).

### 2.6 SIG/BAK/OVRL alohida kesimlar

- NeMo Curator'da SIGMOS filtri misoli: **`4.0` (noise o'lchovi)**, **`3.5` (overall)**. Manba: https://docs.nvidia.com/nemo/curator/main/curate-audio/process-data/quality-filtering/
- Emilia OVRL (`ovrl_mos`) dan foydalanadi, SIG/BAK ni alohida filtrlamaydi.
- SIG va BAK uchun alohida sanoat standart kesimi: **ma'lumot topilmadi**. Amaliy mantiq: BAK past bo'lsa → shovqin muammosi (denoise yordam beradi); SIG past bo'lsa → nutqning o'zi buzilgan (denoise yordam bermaydi, tashlash kerak).

---

## 3. SNR / shovqin bahosi (reference'siz)

### 3.1 WADA-SNR

**LibriTTS asl pipeline'i** (eng ko'p keltiriladigan manba, https://ar5iv.labs.arxiv.org/html/1904.02882):

> "Compute SNR of the audio using waveform amplitude distribution analysis (**WADA**). Audio with **WADA-SNR < 20 dB** and **< 0 dB** are filtered out from the '**clean**' and '**other**' subsets, respectively."

Ta'sir hajmi (train-clean-360, jami 262,107 gap):
- SNR filtri **65,718** gapni o'chirdi
- Alignment muvaffaqiyatsizligi **78,058** gapni chiqardi
- So'z davomiyligi evristikasi yana **26** gapni o'chirdi

Ya'ni LibriTTS'da SNR filtri ~25% ni yo'q qildi.

### 3.2 Boshqa korpuslardagi SNR kesimlari — juda katta tarqalish

| Korpus | SNR threshold | Manba |
|---|---|---|
| LibriTTS "clean" | **≥ 20 dB** (WADA) | https://ar5iv.labs.arxiv.org/html/1904.02882 |
| LibriTTS "other" | **≥ 0 dB** (WADA) | yuqoridagi |
| GLOBE | **≥ 0 dB** + bandwidth **≥ 12 kHz** | https://arxiv.org/html/2406.14875v1 |
| IndicVoices-R | **≥ 25 dB** + **C50 ≥ 30 dB** | https://arxiv.org/html/2409.05356 |
| ParsVoice (natija, filtr emas) | 96.5% segment **> 20 dB**, o'rtacha **41.8 dB** | https://arxiv.org/html/2510.10774 |
| Neyshekar (natija) | median SNR **49.8 dB** (IQR 43.7–56.1), median level **−22.9 dBFS**, median silence ratio **0.30** | https://arxiv.org/html/2609.14542v1 |

> **Ziddiyat:** 0 dB (GLOBE) dan 25 dB (IndicVoices-R) gacha — 25 dB farq. Sababi: GLOBE CommonVoice'dan (in-the-wild), IndicVoices-R studio-sifatga yaqin maqsad qo'ygan. **Bir kishi o'qigan kitob/matn uchun 20 dB (LibriTTS "clean") mantiqiy minimum**, 25 dB esa yaxshi mikrofon bilan erishiladigan maqsad.

### 3.3 Reference'siz o'lchash usullari

1. **WADA-SNR** — waveform amplitude distribution analysis; gamma taqsimot parametri orqali SNR baholaydi, clean reference kerak emas. LibriTTS shu bilan filtrlagan.
2. **MFA hisoblaydigan SNR** — "the ratio of mean speech intensity to mean silence intensity" (alignment natijasida sukunat intervallari ma'lum bo'lgach). Manba: https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/implementations/alignment_analysis.html — **threshold berilmagan**.
3. **VAD asosidagi oddiy baho** — silero VAD segmentlari = speech, qolgani = noise floor; `SNR = 10*log10(RMS_speech^2 / RMS_silence^2)`. Bu MFA usulining VAD varianti. Aniq nashr etilgan threshold: LibriTTS'ning 20 dB ini shu usulga ko'chirib ishlatish odatiy, lekin **aynan VAD-asosidagi SNR uchun nashr etilgan threshold: ma'lumot topilmadi**.
4. **SQUIM SI-SDR** — 16 kHz da SI-SDR bahosi (https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html).
5. **DNSMOS BAK** — fon shovqini sifati bevosita bahosi (PCC 0.98).

---

## 4. Transkript verifikatsiyasi — ASR back-check (CER/WER)

Bu eng muhim filtr: har bir kesilgan segmentni **qaytadan ASR ga berib**, tayinlangan matn bilan solishtirish. Nashr etilgan haqiqiy raqamlar:

| Loyiha / maqola | Qoida | URL |
|---|---|---|
| **Transcribe, Align and Segment** (ukrain tili) | "Segments with a **CER higher than 30%** and a **WER higher than 75%** are discarded." + "we further filter out audio segments with a CER higher than **`cer_threshold=60`** on the edges" (ilk/oxirgi 5 belgi) | https://arxiv.org/html/2406.12674v1 |
| **EuroSpeech** | 3 pog'ona: `CER < 30%` → 61k soat (78.2%); **`CER < 20%` → 51k soat (65.4%, asosiy nashr)**; `CER < 10%` → 32k soat (41.0%). Coarse search "passes the first candidate that has a **CER below 30%**" | https://arxiv.org/html/2510.00514v1 |
| **FASA** (bolalar nutqi aligner) | `σa = 0.1` → **WER < 10% avtomatik qabul**; `σi = 0.3` → **10–30% qo'lda tekshirish navbatiga**; > 30% rad. Post-check: predicted vs aligned uzunlik farqi **> 1 so'z** bo'lsa o'chir | https://arxiv.org/html/2406.17926v1 |
| **GLOBE** | Whisper qayta transkripsiya, WFST normalizatsiya, **edit distance > 1 → o'chir** | https://arxiv.org/html/2406.14875v1 |
| **LibriTTS** | "A sentence is marked as 'aligned' if **all words are matching (with edit-distance of zero)**" | https://ar5iv.labs.arxiv.org/html/1904.02882 |
| **IndicVoices-R** | ikki transkripsiya darajasi orasidagi **CER > 5% → chiqarish** | https://arxiv.org/html/2409.05356 |
| **NeMo Curator** | `WER ≤ 30%` = "high quality"; `WER ≥ 80%` → o'chir; pipeline misolida `WER ≤ 75%` | https://docs.nvidia.com/nemo/curator/main/curate-audio/process-data/quality-assessment/ |
| **WenetSpeech4TTS** | ASR **confidence < 0.7 → chiqarish** (CER emas, model confidence) | https://arxiv.org/html/2406.05763v3 |

### 4.1 Ziddiyat va uni qanday hal qilish

**Manbalar KESKIN ziddiyatda:** LibriTTS/GLOBE `edit distance ≤ 1` (≈ CER 0–2%) talab qiladi, EuroSpeech/Ukraina pipeline'i `CER < 20–30%` ni yetarli deb biladi.

Sabab va yechim:
- **Qattiq mezon (CER ≈ 0, edit distance ≤ 1) = TTS uchun.** TTS modelida matn↔audio mosligi buzilgan bitta segment ham modelni "gallyutsinatsiya" qilishga o'rgatadi. LibriTTS va GLOBE ikkalasi ham TTS korpuslari va ikkalasi ham edit-distance 0–1 ishlatadi.
- **Yumshoq mezon (CER 20–30%, WER 75%) = ASR uchun.** EuroSpeech va Ukraina pipeline'i ASR treningini nishonga oladi; ASR shovqinli labelga chidamli.
- **Chegara alohida:** ilk/oxirgi belgilar uchun yumshoqroq threshold (`edge CER 60`) — chunki kesish nuqtasida ASR hozir kam kontekst ko'radi.

**Yana bir nozik nuqta:** o'z ASR'ingiz bilan back-check qilganda ASR'ning o'z xatosi CER ga qo'shiladi. ParsVoice o'z validatsiya to'plamida ASR (Whisper) ning o'zi **4.90% WER va 1.81% CER** bergani yozilgan (https://arxiv.org/html/2510.10774) — ya'ni "floor" ~2% CER. Shuning uchun **CER threshold ASR floor'idan sezilarli yuqori** bo'lishi kerak; aks holda toza segmentlarni ham rad etasiz. IndicVoices-R ning **CER > 5%** mezoni aynan shu floor ustiga qurilgan.

---

## 5. Alignment confidence filtrlash

### 5.1 ctc-segmentation confidence score

| Manba | Threshold | Izoh |
|---|---|---|
| NeMo `tools/ctc_segmentation/run_segmentation.sh` | **`MIN_SCORE=-2`** | "only those exceeding this confidence score are saved to the final manifest"; `WINDOW ∈ {8000, 12000}` (muvaffaqiyatsizlikda oyna ikki barobar oshadi); `MAX_SEGMENT_LEN=30` (belgi soni); `MAX_DURATION=20` s; `SAMPLE_RATE=16000`. URL: https://github.com/NVIDIA/NeMo/blob/main/tools/ctc_segmentation/run_segmentation.sh |
| NeMo `scripts/cut_audio.py` | `--threshold` default **`-5`** | "Minimum score value accepted"; `--edge_duration` default `0.05` s ("Duration of audio for mean absolute value calculation at the edges"); `--max_duration` default `60` s. URL: https://github.com/NVIDIA/NeMo/blob/main/tools/ctc_segmentation/scripts/cut_audio.py |
| Transcribe-Align-Segment (amalda) | "We retain only audio segments with **alignment confidence score above threshold = −2**" | https://arxiv.org/html/2406.12674v1 |

> **Ziddiyat:** NeMo'ning ikki skripti o'zi ham bir-biriga mos kelmaydi (`-2` vs `-5`). Amaliy talqin: **`-2` qattiq (yuqori sifat), `-5` yumshoq (ko'proq data)**. Skorning ma'nosi — CTC yo'lining normallashtirilgan log-ehtimoli; 0 ga yaqin = ishonchli, manfiy tomon o'sgani sari yomon.
>
> Ushbu skor **absolut emas**: u ASR modelga, tilga va audio sifatiga bog'liq. Shuning uchun amalda: bir marta butun audio uchun skorlarni chiqarib, taqsimotni ko'rib, o'z korpusingizga percentile bo'yicha kesish (masalan eng past 10%) ham ishlatiladi — `-2` esa NeMo tavsiya qilgan absolut boshlang'ich nuqta.

### 5.2 MFA alignment scores

MFA `alignment_analysis.csv` faylida chiqaradigan o'lchovlar (https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/implementations/alignment_analysis.html):

1. **Alignment log-likelihood** — "the objective measure that was optimized for alignment", lekin "**a relative measure** for the best path of alignment for this particular utterance compared to other possible alignments"
2. **Speech log-likelihood** — sukunat intervallarini chiqarib tashlagan holda, "the average of per-phone log-likelihoods in the utterance"
3. **Phone duration deviation** — "the **maximum of the absolute z-score** of each phone's **log-duration**"
4. **Max running short intervals** — ketma-ket **≤ 10 ms** intervallar seriyasi; "long-running stretches of 10ms intervals in a row" → misalignment belgisi
5. **Intensity deviation** — interval intensivligining z-score'i
6. **SNR** — o'rtacha nutq intensivligi / o'rtacha sukunat intensivligi

**MFA uchun "bu segment xato, tashla" degan aniq raqam: `ma'lumot topilmadi`.** Hujjat o'zi tan oladi: muallif "still currently working to refine what are the best measures" va hech qanday absolut kesim bermaydi. Qo'shimcha ogohlantirish (MFA muallifi blogi, https://memcauliffe.com/update-on-montreal-forced-aligner-performance.html): log-likelihood **per-utterance** darajada "missing words" kabi muammolarni topishda muvaffaqiyatli bo'lmagan; u ko'proq butun tizim sifatini baholash uchun foydali.

**Amaliy tavsiya:** MFA skoriga ishonmang. MFA'ni faqat chegaralar uchun ishlating, **rad etish qaroriga esa ASR back-check CER'ini ishlating** (4-bo'lim) — bu tekshirilgan, raqamlari nashr etilgan va talqini aniq.

---

## 6. Speaker verification / diarizatsiya — noto'g'ri ovoz va ustma-ust nutqni tutish

| Tool / loyiha | Cosine similarity threshold | Maqsad | Manba |
|---|---|---|---|
| **SpeechBrain `SpeakerRecognition`** (ECAPA-TDNN VoxCeleb) | **`threshold = 0.25`** (default) — "Threshold applied to the cosine distance to decide if the speaker is different (0) or the same (1)"; `torch.nn.CosineSimilarity(dim=-1, eps=1e-6)` | verifikatsiya (EER'ga kalibrlangan) | https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/inference/speaker.py |
| **WenetSpeech4TTS** | **> 0.65** → qo'shni segmentlarni bir so'zlovchi deb merge | segment merge | https://arxiv.org/html/2406.05763v3 |
| **WenetSpeech4TTS** | **> 0.75** → diarizatsiya klasterlari markazlarini merge | klaster merge | yuqoridagi |
| **ParsVoice** (ECAPA-TDNN) | **0.90** global merge; segment ishonchi **`c_i < 0.4` → chiqarish**; cross-book merge uchun **≥ 100 segment** | juda konservativ klasterlash | https://arxiv.org/html/2510.10774 |
| **Emilia-Pipe** | `pyannote/speaker-diarization-3.1` (threshold pipeline ichida) | diarizatsiya | https://arxiv.org/html/2407.05361v2 |
| **NeMo TitaNet / ECAPA** (o'lchov, threshold emas) | intra-speaker o'rtacha cosine: TitaNet **0.702**, ECAPA-TDNN **0.658** | referens qiymat | https://arxiv.org/pdf/2506.20190 |

> **Ziddiyat sharhi:** 0.25 (SpeechBrain) va 0.90 (ParsVoice) orasidagi farq xato emas — turli maqsad va turli embedding normalizatsiyasi:
> - **0.25** = "ikki utterance bir odamdanmi?" verifikatsiya savoliga EER nuqtasida javob (ko'p noto'g'ri qabul bo'ladi, lekin recall yuqori).
> - **0.65–0.75** = klasterlash/merge qarori (WenetSpeech4TTS).
> - **0.90** = "aniq bir odam" deb ishonch bilan tasdiqlash (ParsVoice, minglab kitob bo'ylab so'zlovchini birlashtirish).
>
> **Bitta so'zlovchi + to'liq matn** stsenariysi uchun (sizning holat) muammo verifikatsiya emas, **anomaliya detektsiyasi**: barcha segmentlardan ECAPA embedding olib, **medoid/o'rtacha embedding** hisoblang, keyin har segmentning shu markazga cosine'ini ko'ring va **eng past 1–5% ni qo'lda ko'rish uchun belgilang**. Bu aynan WenetSpeech4TTS ishlatgan mexanizm: "label the **10%** of embeddings in each cluster that are farthest from the cluster center" (https://arxiv.org/html/2406.05763v3). Absolut kesim uchun `0.65` (WenetSpeech4TTS merge qiymati) xavfsiz minimum sifatida ishlaydi.

**Overlapping speech (ustma-ust nutq):** `pyannote/speaker-diarization-3.1` overlap chiqaradi; Emilia shuni ishlatadi. Aniq "overlap ratio > X% → tashla" nashr etilgan threshold: **ma'lumot topilmadi**. Amaliy: bitta so'zlovchi o'qigan audio'da overlap bo'lmasligi kerak — har qanday overlap segmenti = tashqi ovoz → tashlash.

---

## 7. Loudness va daraja normalizatsiyasi

### 7.1 Nashr etilgan target'lar — ZIDDIYATDA

| Manba | Target | Izoh | URL |
|---|---|---|---|
| **EBU R128** (standart) | **−23 LUFS**, TP **−1 dBTP** | broadcast standarti | (ffmpeg loudnorm hujjatida keltiriladi) |
| **ffmpeg `loudnorm` default** | **`I=-24` LUFS, `TP=-2.0`, `LRA=7`** | ATSC A/85 (US broadcast) | https://ayosec.github.io/ffmpeg-filters-docs/7.1/Filters/Audio/loudnorm.html |
| **TTS-DATA-PIPELINE** (amaliy repo) | **−23 LUFS** (faqat qabul qilingan kliplar; rad etilganlar normalizatsiya qilinmaydi) | TTS dataset pipeline | https://github.com/harika1903/TTS-DATA-PIPELINE |
| **Emilia-Pipe** | **`target_dBFS = -20`**, gain **±3 dB** bilan cheklangan | LUFS emas, dBFS (RMS) | https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/main.py |
| **Google Assistant** (TTS chiqishi) | stereo **−16 LUFS**, mono **−19 LUFS** | dataset emas, playback | https://developers.google.com/assistant/tools/audio-loudness |
| **Neyshekar** (o'lchov natijasi) | median **−22.9 dBFS** | filtr emas, statistika | https://arxiv.org/html/2609.14542v1 |

> **Ziddiyat:** −16, −19, −20, −23, −24 — beshta turli raqam. Ammo ular turli narsani o'lchaydi: **−20 dBFS ≠ −20 LUFS** (dBFS = RMS, LUFS = K-weighted, gating bilan). TTS dataset uchun eng ko'p uchraydigani: **−23 LUFS (EBU R128)** yoki Emilia uslubi **−20 dBFS ± 3 dB cheklov bilan**.
>
> **Muhim nozik jihat (Emilia'dan o'rganish kerak):** Emilia gain'ni **±3 dB** bilan cheklaydi. Sababi — agar segmentda asosan sukunat bo'lsa, RMS past chiqadi va cheksiz gain shovqinni kuchaytiradi. **Har doim gain'ga limit qo'ying.**

### 7.2 pyloudnorm

- ITU-R BS.1770-4 ni amalga oshiradi. Gating block default **400 ms** (`block_size=0.200` bilan o'zgartirish mumkin). Manba: https://github.com/csteinmetz1/pyloudnorm

```python
import soundfile as sf, pyloudnorm as pyln
data, rate = sf.read("seg.wav")
meter = pyln.Meter(rate)              # BS.1770-4, 400 ms block
loudness = meter.integrated_loudness(data)
out = pyln.normalize.loudness(data, loudness, -23.0)
```

**Diqqat:** 400 ms gating block'i borligi uchun **juda qisqa segmentlarda** (< ~0.5 s) integrated loudness ishonchsiz. pyloudnorm README'da minimal davomiylik talabi aniq yozilmagan — **`ma'lumot topilmadi`**. Amaliy yechim: loudness'ni **butun master audio** bo'yicha bir marta o'lchab, **bitta global gain** qo'llash — segmentlar orasidagi darajani ham saqlaydi, qisqa segment muammosini ham yo'q qiladi.

### 7.3 ffmpeg loudnorm — two-pass

- 1-o'tish: `-af loudnorm=I=-23:TP=-1:LRA=7:print_format=json -f null -` → o'lchangan `measured_I`, `measured_TP`, `measured_LRA`, `measured_thresh`, `target_offset` ni oladi.
- 2-o'tish: o'sha qiymatlarni `measured_*` parametrlarga qaytarib, **`linear=true`** bilan → butun faylga **bitta doimiy gain** qo'llanadi. `linear=false` (default, dynamic) esa LRA maqsadiga sig'dirish uchun **audio'ni kompressiya qiladi** — bu TTS dataset uchun **zararli** (dinamik diapazonni buzadi, modelga noto'g'ri prosodiya o'rgatadi). Manba: https://ayosec.github.io/ffmpeg-filters-docs/7.1/Filters/Audio/loudnorm.html va https://dev.to/masonwritescode/two-pass-loudness-normalization-with-ffmpeg-loudnorm-the-right-way-1nm3

> **TTS dataset uchun qoida: har doim `linear=true` (yoki pyloudnorm + qo'lda gain). Dynamic mode = kompressiya = yo'q.**

### 7.4 Peak normalize vs loudness normalize

- Peak normalize (masalan `-1 dBFS` ga) darajani **tenglashtirmaydi** — bitta baland qichqiriq butun faylni pasaytiradi.
- TTS dataset uchun to'g'ri tartib: **(1) loudness normalize → (2) true-peak limiter/clip check (`TP ≤ -1 dBTP`)**. ffmpeg loudnorm ikkalasini birga qiladi (`I` + `TP`).

### 7.5 DC offset

- LibriTTS aynan shu qadamni qo'yadi: "**Normalize the polarity of audio by flipping up-side-down waveforms by ensuring DC offsets are positive**" (https://ar5iv.labs.arxiv.org/html/1904.02882). Ya'ni ikki ish: (a) polarity/faza tekshiruvi, (b) DC offset bilan hisob-kitob.
- Amaliy: `dc = mean(x)`; agar `|dc| > ~0.001` (≈ −60 dBFS) bo'lsa `x -= dc` yoki high-pass filter (20–30 Hz). **Aniq nashr etilgan DC offset kesim raqami: `ma'lumot topilmadi`** — LibriTTS raqam bermaydi, faqat protsedurani aytadi.
- DC offset clipping detektsiyasini ham buzadi: "One limitation of clipping detection algorithms are potential false negatives when a signal bias (DC offset) is introduced" — ya'ni **DC offset'ni clipping tekshiruvidan OLDIN olib tashlash kerak**. (Manba: https://www.sciencedirect.com/science/article/pii/S0167639321000832)

### 7.6 Clipping detektsiyasi

| Mezon | Qiymat | Manba |
|---|---|---|
| "Flattened run" ta'rifi | **≥ 3 ketma-ket full-scale sample** | https://arxiv.org/html/2609.14542v1 |
| Qattiq mezon (korpus statistikasi) | samplelarning **> 0.1%** rail'da → muammoli klip | yuqoridagi (0.62% klip shu holatda) |
| Ball tizimi jazosi | clipping **> 5% → −20 ball** | https://arxiv.org/html/2510.10774 (ParsVoice) |
| Eshitilish chegarasi | "as little as **0.01%** clipping can be displeasing to the audio experience" | https://www.sciencedirect.com/science/article/pii/S0167639321000832 (va sohaviy adabiyot) |

> **Ziddiyat:** 0.01% (eshitiladi) vs 0.1% (korpus mezoni) vs 5% (ParsVoice jazo chegarasi). TTS dataset uchun **0.1%** amaliy kesim: undan yuqorisi → tashla yoki declip.

### 7.7 Dinamik diapazon

- ffmpeg loudnorm `LRA` (Loudness Range) ni o'lchaydi; default target `7` LU.
- TTS dataset uchun tavsiya etilgan LRA qiymati: **`ma'lumot topilmadi`**. Faqat ma'lum bo'lgani: `linear=true` bilan LRA o'zgartirilmaydi (kompressiya yo'q) — bu TTS uchun afzal.
- ParsVoice o'rniga "silence ratio" ni ishlatadi: **> 50% → −15 ball**, **30–50% → −10 ball** (https://arxiv.org/html/2510.10774). Neyshekar median silence ratio **0.30** (https://arxiv.org/html/2609.14542v1).

---

## 8. Sample rate / format

### 8.1 Qaysi model uchun qaysi rate

| Model / korpus | Sample rate | Manba |
|---|---|---|
| **Coqui TTS umumiy tavsiya** | "It is better to reduce the sample-rate of your dataset to around **16000–22050**" | https://docs.coqui.ai/en/latest/what_makes_a_good_dataset.html |
| **VITS / LJSpeech an'anaviy setup** | **22050 Hz** (LJSpeech 22.05 kHz; VCTK 44.1 kHz ni ham 22.05 kHz ga tushiradilar) | https://docs.coqui.ai/en/latest/models/vits.html |
| **LibriTTS / LibriTTS-R** | **24 kHz** ("The audio files are at 24kHz sampling rate") | https://ar5iv.labs.arxiv.org/html/1904.02882 |
| **Emilia** | **24 kHz**, 16-bit (`set_sample_width(2)`) | https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/main.py |
| **WenetSpeech4TTS** | "resample all recordings to **24k Hz**" | https://arxiv.org/html/2406.05763v3 |
| **GLOBE** | **24 kHz** (asl 48 kHz dan) | https://arxiv.org/html/2406.14875v1 |
| **StyleTTS2** | **24 kHz** | https://github.com/yl4579/StyleTTS2/discussions/81 |
| **Neyshekar** | **16 kHz** mono, 16-bit PCM WAV | https://arxiv.org/html/2609.14542v1 |

**Vositalarning majburiy rate'lari (bularni chetlab o'tolmaysiz):**

| Vosita | Talab | Manba |
|---|---|---|
| silero-vad | faqat **8000 / 16000 Hz** | https://github.com/snakers4/silero-vad |
| DNSMOS (speechmos) | **16000 Hz** | https://pypi.org/project/speechmos/ |
| torchaudio SQUIM | "only supports **16000 Hz**" | https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html |
| UTMOS (`utmos22_strong`) | `sample_rate=16000` — "Do not change unless you have a custom checkpoint" | https://docs.nvidia.com/nemo/curator/main/curate-audio/process-data/quality-filtering/utmos |
| DeepFilterNet | "only wav files with a sampling rate of **48kHz** are supported" | https://github.com/Rikorose/DeepFilterNet |
| resemble-enhance | 44.1 kHz sifatli nutqda o'qitilgan (chiqish 44.1 kHz) | https://github.com/resemble-ai/resemble-enhance |
| NeMo ctc-segmentation | `SAMPLE_RATE=16000` | https://github.com/NVIDIA/NeMo/blob/main/tools/ctc_segmentation/run_segmentation.sh |

### 8.2 Tavsiya etilgan arxitektura: master + derivatives

Yuqoridagi jadvaldan kelib chiqadigan yagona mantiqiy yechim:

1. **Master (arxiv):** asl native rate'da (48 kHz yoki 44.1 kHz), **mono**, **24-bit yoki 16-bit PCM WAV**, loudness-normalized, DC-offset tozalangan. Hech qachon o'chirilmaydi.
2. **Analysis kopiyasi:** 16 kHz mono — VAD, DNSMOS, UTMOS, SQUIM, ASR, ctc-segmentation uchun.
3. **Enhancement kopiyasi:** 48 kHz — DeepFilterNet uchun (agar ishlatilsa).
4. **Model export:** 22050 Hz (VITS/LJSpeech-style) va/yoki 24000 Hz (LibriTTS/XTTS/StyleTTS2-style) — masterdan bitta qadamda `soxr_vhq` bilan.

**Kritik qoida:** hech qachon **ketma-ket resample** qilmang (48k → 16k → 22.05k). Har doim **masterdan bitta qadamda** target rate'ga o'tkazing, aks holda anti-alias filtrlar ustma-ust tushib yuqori chastotalarni yo'qotadi.

### 8.3 Resampling sifati

`librosa.resample` `res_type` (https://librosa.org/doc/latest/generated/librosa.resample.html):

| `res_type` | Tavsif |
|---|---|
| **`soxr_vhq`** | Very high quality, FFT-based bandlimited interpolation — **dataset ishlab chiqarish uchun** |
| **`soxr_hq`** | **default** — high quality bandlimited |
| `soxr_mq`, `soxr_lq` | medium / low quality |
| `soxr_qq` | "Quick cubic interpolation (very fast, but **not bandlimited**)" |
| `kaiser_best`, `kaiser_fast` | resampy orqali |
| `fft` / `scipy` | Fourier-based |
| `polyphase` | tez, faqat butun sample-rate'lar |
| `linear`, `zero_order_hold` | juda tez, sifat past |

Rasmiy ogohlantirish: **`soxr_qq`, `polyphase`, `linear`, `zero_order_hold` aliasing kiritishi mumkin.** Dataset uchun faqat `soxr_vhq` yoki `soxr_hq` ishlating. `soxr` `scipy.signal.resample` dan ham tezroq va aniqroq (10 s 48 kHz → 44.1 kHz: soxr HQ ~7.2 ms vs scipy ~13.4 ms — https://pypi.org/project/soxr/).

### 8.4 Xato qilsangiz nima bo'ladi

| Xato | Natija |
|---|---|
| Datasetda turli sample rate aralashib ketsa | spektral artefaktlar → "unnatural prosody"; mel-spektrogramma o'lchamlari mos kelmaydi (manba sifatida: https://docs.coqui.ai/en/latest/what_makes_a_good_dataset.html — "Background noise might lead your model to struggle, especially for a good alignment"; sample-rate aralashuvi bo'yicha aniq nashr etilgan miqdoriy zarar: **ma'lumot topilmadi**) |
| Aliasing beruvchi resampler (`polyphase`, `linear`) | yuqori chastotada oyna (mirror) chastotalar → vocoder shovqinli chiqadi |
| 16 kHz masterdan 22.05/24 kHz "upsample" | 8 kHz dan yuqorida hech narsa yo'q → model "o'lik" yuqori band o'rganadi, jarangsiz undoshlar (s, sh, f) sifatsiz |
| Stereo ni noto'g'ri mono qilish (bitta kanalni tashlash) | agar kanallar faza teskari bo'lsa, `(L+R)/2` nutqni yo'q qiladi → LibriTTS polarity tekshiruvi shu uchun kerak |
| Bit depth 8-bit yoki float32 aralash | 8-bit da kvantlash shovqini eshitiladi; float32 WAV ba'zi loader'larda noto'g'ri o'qiladi. Standart: **16-bit PCM** (Emilia `set_sample_width(2)`) |

---

## 9. Denoise / dereverb / enhancement — QILISH KERAKMI?

**Qisqa javob: HA, lekin tekshiruv bilan.** To'rtta mustaqil manba enhancement TTS sifatini oshirganini ko'rsatadi, faqat bittasi "ehtiyot bo'l" deydi.

### 9.1 LibriTTS-R — eng kuchli dalil (bevosita so'ralgan)

**Nima qilindi:** LibriTTS ga **Miipher** speech restoration modeli qo'llanildi (w2v-BERT feature + DF-Conformer feature cleaner + PnG-BERT matn conditioning + **WaveFit-5** vocoder). Manba: https://ar5iv.labs.arxiv.org/html/2305.18802 , https://www.isca-archive.org/interspeech_2023/koizumi23_interspeech.html

**Ground-truth audio MOS:**

| Split | LibriTTS | LibriTTS-R |
|---|---|---|
| test-clean | **4.36 ± 0.08** | **4.41 ± 0.07** |
| test-other | **3.94 ± 0.10** | **4.09 ± 0.10** |

Side-by-side preference (LibriTTS-R foydasiga): test-clean **+0.80 ± 0.15**, test-other **+1.42 ± 0.14**.

**Eng muhim jadval — train-960 da o'qitilgan TTS MOS:**

| So'zlovchi | LibriTTS train-960 | **LibriTTS-R train-960** |
|---|---|---|
| 19 | 2.59 ± 0.10 | **4.06 ± 0.08** |
| 103 | 2.75 ± 0.10 | **4.31 ± 0.08** |
| 1841 | 3.35 ± 0.10 | **4.20 ± 0.08** |
| 204 | 2.74 ± 0.09 | **4.11 ± 0.08** |
| 1121 | 2.83 ± 0.10 | **4.23 ± 0.07** |
| 5717 | 2.97 ± 0.10 | **4.08 ± 0.08** |

> **+1.3 dan +1.6 MOS gacha o'sish.** Bu TTS tadqiqotida g'oyat katta farq. Ground-truth audio bahosi faqat +0.05…+0.15 oshganiga qaramay, **TTS treningida farq bir necha barobar kattaroq** chiqdi — chunki model shovqinni ham o'rganib oladi va uni sintezda takrorlaydi.

**Muallif ogohlantirishi (juda muhim):**
> "we have found a few examples that LibriTTS speech sample achieved a better score in SxS comparison. By listening these examples, **two of 640 LibriTTS-R speech samples were distorted due to the failure of SR**."
> "Since it is difficult to manually check all samples, we have not checked all speech samples in LibriTTS-R. Therefore, **the samples in training splits may also contain a small number of distorted samples**."

Ya'ni **~0.3% (2/640) restoration muvaffaqiyatsizligi** bo'ldi va u aniqlanmagan holda dataset ichida qoldi. **Xulosa: enhancement'dan keyin FILTRLASH shart.**

### 9.2 Boshqa dalillar

| Maqola | Modellar | Natija | Manba |
|---|---|---|---|
| **Exploring Speech Enhancement for Low-resource Speech Synthesis** | DEMUCS, FullSubNet, FRCRN, D2Former, **TF-GridNet** | WER: reference (clean) **18.6%**, unprocessed **57.6%**, re-synthesis 57.4%, DEMUCS **46.6%**, FullSubNet 47.1%, FRCRN 46.2%, D2Former 50.6%, **TF-GridNet 44.7%** (−12.9 absolut). Ogohlantirish: "**high speech enhancement performance doesn't always align with T2U model performance**" | https://arxiv.org/html/2309.10795v1 |
| **Enhancing Crowdsourced Audio for TTS** | **VoiceFixer** + SoX spectral subtraction | UTMOS **2.54 ± 0.06 → 2.94 ± 0.05** (+0.40), PESQ 3.19 → 3.38, STOI 0.95 → 0.98, SI-SDR 20.14 → 22.61. Eng yaxshi 100 h subset **NISQA ≥ 4.0** filtri bilan | https://arxiv.org/html/2410.13357v1 |
| **Evaluation of preprocessing pipelines (in-the-wild TTS)** | Demucs vs boshqalar, NISQA/DNSMOS filtrlari | **Eng yaxshi konfiguratsiya: Demucs denoising + DNSMOS threshold 2.7.** "denoising variants with **permissive filtering** provide the best overall compromise"; qattiq filtr "reduced dataset size without proportional quality gains". Ovoz saqlanishi mezoni: **MCD ≤ 5 dB** | https://arxiv.org/html/2510.03111 |
| **Confidence-based Filtering + Generative SE** | Genhancer (discrete token GSE) + confidence filtr | noisy → unfiltered enhanced → **filtered enhanced**: UTMOS **2.73 → 3.64 → 3.80**; DNSMOS **2.74 → 3.10 → 3.17**; WER **21.31% → 20.45% → 18.79%**. Filtrlash percentile bilan: top **90/80/70/60/50%**; **top 80% eng yaxshi** | https://arxiv.org/html/2601.12254 |
| **IndicVoices-R** | HTDemucs (separation/denoise) + **VoiceFixer (dereverb)** + **DeepFilterNet3** (full-band SE) | "although pre-trained on English, show good **cross-lingual generalization**" — o'zbek tili uchun ham dalil | https://arxiv.org/html/2409.05356 |

### 9.3 Xulosa: qanday qilish

1. **Denoise QIL** — 5 mustaqil manba foydasini tasdiqlaydi. Eng ko'p tanlanadigani: **Demucs / HTDemucs** (in-the-wild eval eng yaxshi deb topdi) yoki **DeepFilterNet3** (Windows'da eng oson).
2. **Dereverb alohida** — VoiceFixer (IndicVoices-R shuni ishlatadi). Reverb darajasini **C50 ≥ 30 dB** bilan o'lchash (IndicVoices-R mezoni).
3. **Enhancement'dan KEYIN qayta filtrlash SHART** — LibriTTS-R 0.3% buzilgan sample'ni o'tkazib yubordi. Filtr: DNSMOS/UTMOS qayta hisoblash + **MCD ≤ 5 dB** (original vs enhanced) ovoz o'zgarmaganini tasdiqlash.
4. **Yumshoq filtr + denoise > qattiq filtr denoise'siz** (in-the-wild eval xulosasi). Kichik dataset uchun bu ayniqsa muhim.
5. **Originalni saqlang.** Enhancement qaytarilmaydi; A/B qiyoslash uchun original master kerak.
6. **Ogohlantirish:** enhancement metrikasi yaxshilanishi TTS yaxshilanishini kafolatlamaydi ("high speech enhancement performance doesn't always align with T2U model performance") — yakuniy qaror TTS namunalarini eshitib qabul qilinishi kerak.

---

## 10. Sukunat kesish (silence trim) va nomutanosiblik muammosi

### 10.1 Muammoning mexanizmi

- **StyleTTS2 finetuning notes** (https://github.com/yl4579/StyleTTS2/discussions/81): datasetda ko'p yetakchi/oxirgi sukunat bo'lsa, model shuni o'rganadi va generatsiya oxirida artefakt chiqaradi. Yechim: **barcha fayl oxiriga 100 ms sukunat + stop-token** qo'shish. Alternativ: `$` token = **5000 sample ≈ 208 ms @ 24 kHz**, transkriptning boshiga ham oxiriga ham qo'yish mumkin.
- **Frame-Wise Breath Detection** (https://arxiv.org/html/2402.00288v1): baseline modellar sukunatning bir qismini nafas (breath) deb xato aniqlaydi → TTS sukunatni nafas sifatida talqin qiladi va sintezda nafas/shovqin chiqaradi.
- Umumiy mexanizm: **past darajali fon shovqini o'rganiladigan pattern** bo'lib qoladi; model sukunat = "shitirlash" degan qoidani o'rganadi.

### 10.2 Amaliy qoida (manbalardan)

| Qoida | Qiymat | Manba |
|---|---|---|
| Nolga kesish + **fiksirlangan** pad qo'shish | oxiriga **100 ms** sukunat (+ stop token) | https://github.com/yl4579/StyleTTS2/discussions/81 |
| Muqobil fiksirlangan pad | **208 ms** (5000 sample @ 24 kHz), boshiga ham oxiriga ham | yuqoridagi |
| VAD chegara padding | **200–300 ms** (`speech_pad_ms=300` NeMo default, TTS uchun 400–500) | https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-filtering/vad |
| Ichki sukunat limiti | **> 930 ms** uzluksiz ichki sukunat → utterance'ni o'chirish | https://arxiv.org/html/2406.14875v1 |
| Chegara optimallashtirish | chegaradan **max 3 s** kesish, **0.1 s** qadam, **max 5 s** kengaytirish (binary search) | https://arxiv.org/html/2510.10774 |
| Silence ratio | **> 50% → −15 ball**, **30–50% → −10 ball** | https://arxiv.org/html/2510.10774 |
| Uzun bosh/oxir sukunatni olib tashlash | "Run a silence end-pointer to remove long start and end silences" | https://ar5iv.labs.arxiv.org/html/1904.02882 |

### 10.3 Eng muhim printsip: MUTANOSIBLIK

Manbalardan chiqadigan asosiy xulosa — **aniq qancha pad qo'shganingiz emas, HAR BIR faylda BIR XIL qo'shganingiz muhim.** Retsept:

1. VAD bilan nutq chegaralarini top.
2. Chegaralarni **nolga qadar kes** (barcha sukunatni olib tashla).
3. Keyin **har bir faylga bir xil miqdorda** sukunat qo'sh: masalan boshiga **100 ms**, oxiriga **100 ms** (yoki StyleTTS2 uslubida faqat oxiriga 100 ms).
4. Qo'shiladigan sukunat **haqiqiy nol bo'lmasin** — agar model treningda mutlaq nol ko'rib, inference'da noise floor kutsa nomutanosiblik chiqadi. **Amaliy yechim: shu faylning o'z noise floor'idan olingan sukunat qismini ishlatish** (nol emas, tabiiy jim qism). Bu bo'yicha aniq nashr etilgan tavsiya: **`ma'lumot topilmadi`** — StyleTTS2 muhokamasi "silence" deydi, turini aniqlamaydi.
5. Ichki (segment o'rtasidagi) pauzalarni **kesmang** — ular prosodiya. Faqat **> 930 ms** dan uzunlari bo'lsa (GLOBE mezoni) shu segmentni ikkiga bo'ling yoki tashlang.

---

## 11. Windows 11 mosligi (GPU bilan)

### 11.1 Windows'da pip bilan muammosiz ishlaydiganlar

| Tool | Izoh |
|---|---|
| **silero-vad** | pip, ONNX/torch — muammosiz |
| **py-webrtcvad** | pip wheel mavjud (ba'zan `webrtcvad-wheels` kerak) |
| **auditok** | sof Python |
| **audio-slicer (openvpi)** | sof Python (numpy + soundfile) |
| **librosa + soxr** | pip wheel Windows uchun bor |
| **pyloudnorm** | sof Python |
| **ffmpeg** | `winget install ffmpeg` yoki gyan.dev build |
| **speechmos** (DNSMOS P.835, PLCMOS, AECMOS) | pip, ONNX Runtime — Windows'da eng oson DNSMOS yo'li. https://pypi.org/project/speechmos/ |
| **torchaudio SQUIM** | torchaudio ichida, CUDA bilan ishlaydi |
| **NISQA** | PyTorch, sof Python. https://github.com/gabrielmittag/NISQA |
| **pyannote.audio** | pip, HuggingFace token kerak (gated modellar) |
| **SpeechBrain** (ECAPA-TDNN) | pip, Windows'da ishlaydi |
| **Demucs / HTDemucs** | pip `demucs`, CUDA bilan ishlaydi |
| **DeepFilterNet** | **rasmiy:** "This framework supports **Linux, MacOS and Windows**. Training is only tested under Linux." `pip install deepfilternet` — Rust'ni qo'lda kompilatsiya qilish shart emas. https://github.com/Rikorose/DeepFilterNet |
| **faster-whisper / WhisperX** | CTranslate2 wheel Windows uchun bor (cuDNN kerak) |
| **VoiceFixer** | pip, sof PyTorch |

### 11.2 Windows'da MUAMMOLI — WSL yoki maxsus qadam kerak

| Tool | Muammo | Yechim |
|---|---|---|
| **ctc-segmentation** | PyPI'da **faqat sdist** (`ctc_segmentation-1.7.4.tar.gz`), Windows wheel yo'q → Cython/C kompilyator kerak. https://pypi.org/project/ctc-segmentation/ | MSVC Build Tools o'rnatish, yoki WSL2 |
| **resemble-enhance** | `deepspeed` talab qiladi → Windows'da `LINK : fatal error LNK1181: cannot open input file 'aio.lib'`. GitHub issue #1 va #60 | Tayyor wheel (resemble-enhance-windows releases, Python 3.10 + CUDA 11.8/12.1) yoki **WSL2**. https://github.com/resemble-ai/resemble-enhance/issues/1 |
| **NeMo toolkit** (text normalization, ctc_segmentation tools) | `pynini`/`openfst` — Windows'da sof pip bilan o'rnatilmaydi | conda-forge yoki **WSL2** |
| **Montreal Forced Aligner** | conda-forge bilan Windows'da o'rnatiladi: `conda create -n aligner -c conda-forge montreal-forced-aligner -y`. Ammo hujjatda "If you're on **Mac or Linux** and want G2P functionality, install OpenFst, OpenGrm-NGram, Baum-Welch, and Pynini" → **G2P funksiyasi Windows'da cheklangan bo'lishi mumkin**. Test suite `tox -e py38-win` mavjud, ya'ni Windows sinovdan o'tadi. https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html | conda (ishlaydi) yoki Docker/WSL2 (G2P uchun) |
| **UTMOS asl repo** (sarulab-speech/UTMOS22) | `fairseq` talab qiladi → Windows'da og'riqli | `utmos22_strong` checkpoint'ini torch.hub orqali (NeMo Curator shu checkpoint'ni ishlatadi) yoki `utmosv2` |

### 11.3 Windows uchun tavsiya etilgan stack

Barcha kerakli filtrlarni **WSL'siz** yopish mumkin:

```
Kesish:        silero-vad (pip)
Sifat:         speechmos (DNSMOS P.835) + torchaudio SQUIM + NISQA
               [UTMOS: utmos22_strong torch.hub]
SNR:           WADA-SNR (o'z implementatsiyasi, ~50 satr numpy) + DNSMOS BAK
ASR back-check: faster-whisper (CUDA)
Speaker:       SpeechBrain ECAPA-TDNN
Loudness:      pyloudnorm + ffmpeg loudnorm (two-pass, linear=true)
Resample:      soxr (soxr_vhq)
Denoise:       DeepFilterNet3 (48 kHz) yoki Demucs
Alignment:     [boshqa agent lanesida] — lekin ctc-segmentation uchun
               MSVC Build Tools yoki WSL2 kerakligini hisobga oling
```

---

## 12. Nima topilmadi (`ma'lumot topilmadi`)

Bu ro'yxat muhim: bu raqamlarni **o'zingiz eksperiment bilan aniqlashingiz kerak**, chunki nashr etilgan qiymat yo'q.

| Savol | Holat |
|---|---|
| MFA alignment log-likelihood uchun "tashla" thresholdi | **ma'lumot topilmadi** — rasmiy hujjat "still currently working to refine what are the best measures" deydi |
| MFA phone duration deviation z-score kesimi | **ma'lumot topilmadi** |
| SQUIM STOI/PESQ/SI-SDR uchun accept/reject kesimi | **ma'lumot topilmadi** (IndicVoices-R ularni "aggregate" qiladi, alohida raqam yo'q) |
| WV-MOS shkalasi va kesimi | **ma'lumot topilmadi** |
| SIG va BAK uchun alohida sanoat kesimi (OVRL dan tashqari) | **ma'lumot topilmadi** |
| VAD-asosidagi SNR (WADA emas) uchun nashr etilgan kesim | **ma'lumot topilmadi** |
| DC offset uchun aniq raqamli kesim | **ma'lumot topilmadi** (LibriTTS protsedurani aytadi, raqam bermaydi) |
| TTS dataset uchun tavsiya etilgan LRA (dinamik diapazon) qiymati | **ma'lumot topilmadi** |
| pyloudnorm minimal audio davomiyligi | **ma'lumot topilmadi** (faqat 400 ms gating block ma'lum) |
| Overlapping speech ratio kesimi ("overlap > X% → tashla") | **ma'lumot topilmadi** |
| Sukunat pad'i uchun "nol vs tabiiy noise floor" tavsiyasi | **ma'lumot topilmadi** |
| Sample-rate aralashuvi zararining miqdoriy o'lchovi | **ma'lumot topilmadi** |
| `neg_threshold` (silero hysteresis) uchun tavsiya etilgan qiymat | **ma'lumot topilmadi** (parametr bor, tavsiya yo'q) |

---

## 13. Manbalar (to'liq ro'yxat)

**Papers / korpuslar**
- Emilia / Emilia-Pipe: https://arxiv.org/html/2407.05361v2
- LibriTTS-R (Miipher): https://ar5iv.labs.arxiv.org/html/2305.18802 , https://www.isca-archive.org/interspeech_2023/koizumi23_interspeech.html
- LibriTTS: https://ar5iv.labs.arxiv.org/html/1904.02882
- WenetSpeech4TTS: https://arxiv.org/html/2406.05763v3
- GLOBE: https://arxiv.org/html/2406.14875v1
- IndicVoices-R: https://arxiv.org/html/2409.05356
- ParsVoice: https://arxiv.org/html/2510.10774
- EuroSpeech: https://arxiv.org/html/2510.00514v1
- Transcribe, Align and Segment: https://arxiv.org/html/2406.12674v1
- FASA: https://arxiv.org/html/2406.17926v1
- Evaluation of preprocessing pipelines (in-the-wild TTS): https://arxiv.org/html/2510.03111
- Exploring Speech Enhancement for Low-resource Speech Synthesis: https://arxiv.org/html/2309.10795v1
- Enhancing Crowdsourced Audio for TTS: https://arxiv.org/html/2410.13357v1
- Confidence-based Filtering + Generative SE: https://arxiv.org/html/2601.12254
- Neyshekar (Persian read-speech): https://arxiv.org/html/2609.14542v1
- DNSMOS P.835: https://arxiv.org/abs/2110.01763
- Frame-Wise Breath Detection: https://arxiv.org/html/2402.00288v1
- ECAPA/x-vector in zero-shot TTS: https://arxiv.org/pdf/2506.20190
- Nonlinear waveform distortion / clipping: https://www.sciencedirect.com/science/article/pii/S0167639321000832

**Repos / hujjatlar**
- silero-vad: https://github.com/snakers4/silero-vad , https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
- py-webrtcvad: https://github.com/wiseman/py-webrtcvad
- auditok: https://github.com/amsehili/auditok
- audio-slicer: https://github.com/openvpi/audio-slicer/blob/main/slicer2.py
- pyannote/segmentation: https://huggingface.co/pyannote/segmentation
- Emilia-Pipe kodi: https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/main.py
- NeMo ctc_segmentation: https://github.com/NVIDIA/NeMo/blob/main/tools/ctc_segmentation/run_segmentation.sh , https://github.com/NVIDIA/NeMo/blob/main/tools/ctc_segmentation/scripts/cut_audio.py
- NeMo Curator UTMOS filter: https://docs.nvidia.com/nemo/curator/main/curate-audio/process-data/quality-filtering/utmos
- NeMo Curator VAD: https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-filtering/vad
- NeMo Curator quality assessment (WER): https://docs.nvidia.com/nemo/curator/main/curate-audio/process-data/quality-assessment/
- MFA alignment analysis: https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/implementations/alignment_analysis.html
- MFA installation: https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
- MFA performance blog: https://memcauliffe.com/update-on-montreal-forced-aligner-performance.html
- SpeechBrain speaker inference: https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/inference/speaker.py
- NISQA: https://github.com/gabrielmittag/NISQA
- speechmos (DNSMOS pip): https://pypi.org/project/speechmos/
- DNS-Challenge DNSMOS: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
- UTMOS22: https://github.com/sarulab-speech/UTMOS22
- WV-MOS: https://github.com/AndreevP/wvmos
- torchaudio SQUIM: https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html
- pyloudnorm: https://github.com/csteinmetz1/pyloudnorm
- ffmpeg loudnorm: https://ayosec.github.io/ffmpeg-filters-docs/7.1/Filters/Audio/loudnorm.html
- ffmpeg two-pass loudnorm: https://dev.to/masonwritescode/two-pass-loudness-normalization-with-ffmpeg-loudnorm-the-right-way-1nm3
- librosa.resample: https://librosa.org/doc/latest/generated/librosa.resample.html
- soxr: https://pypi.org/project/soxr/
- Coqui TTS dataset guide: https://docs.coqui.ai/en/latest/what_makes_a_good_dataset.html
- Coqui VITS: https://docs.coqui.ai/en/latest/models/vits.html
- DeepFilterNet: https://github.com/Rikorose/DeepFilterNet
- resemble-enhance: https://github.com/resemble-ai/resemble-enhance , https://github.com/resemble-ai/resemble-enhance/issues/1
- ctc-segmentation PyPI: https://pypi.org/project/ctc-segmentation/
- StyleTTS2 finetuning notes: https://github.com/yl4579/StyleTTS2/discussions/81
- TTS-DATA-PIPELINE (-23 LUFS): https://github.com/harika1903/TTS-DATA-PIPELINE
- Google Assistant audio loudness: https://developers.google.com/assistant/tools/audio-loudness

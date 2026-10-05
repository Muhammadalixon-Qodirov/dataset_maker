# Zero-shot TTS / ovoz klonlash: KIM qilgan va QANDAY raqamlar bilan

**Tuzilgan sana:** 2026-10-01
**Qamrov:** ~10 sekundlik referens audiodan ovozni klonlovchi tizimlarning to'liq inventarizatsiyasi — tashkilot, yil, arxitektura oilasi, trening soati, spiker/til soni, parametr, trening GPU, ochiq vazn, **litsenziya** va **kerakli referens audio uzunligi**.

> **Metodologiya va halollik eslatmasi.** Har bir katak faqat paper / HuggingFace model kartasi / repo LICENSE faylidan tasdiqlangan bo'lsa yozilgan. Tasdiqlanmagan hamma narsa **`ma'lumot topilmadi`** deb belgilangan — taxmin qilinmagan. Manbalar bir-biriga zid bo'lgan joyda **ikkalasi ham** keltirilgan va zidlik aytilgan. Litsenziya va trening soati — eng ko'p xato xabar qilinadigan ikki ustun, shuning uchun ular alohida tekshirilgan (quyida "Litsenziya tuzoqlari" bo'limiga qarang).
>
> **Qidiruv budjeti:** 18 WebSearch ruxsat etilgan edi, 6 tasi ishlatildi; qolgani HuggingFace model kartalari, arXiv HTML'lari va GitHub LICENSE fayllarini to'g'ridan-to'g'ri o'qish orqali bajarildi. **Ikki manba olinmadi:** TTS Arena V2 (HuggingFace Space, HTTP 401) va `microsoft/VibeVoice-Large` model kartasi (HTTP 401). Bu yerda ular bo'yicha raqam yo'q.

---

## 1. ASOSIY JADVAL — barcha tizimlar

Ustunlar: Model | Tashkilot | Yil | Arxitektura oilasi | Trening soati | Spiker soni | Til soni | Parametr | Trening GPU | Ochiq vazn | Litsenziya (tijorat?) | Referens audio | Link

### 1a. Akademik / tadqiqot nasl-nasabi (vazni YOPIQ — faqat paper)

| Model | Tashkilot | Yil | Arxitektura oilasi | Trening soati | Spiker soni | Til soni | Parametr | Trening GPU | Ochiq vazn | Litsenziya (tijorat?) | Referens audio | Link |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SV2TTS** (Jia et al.) | Google | 2018 | speaker-embedding (speaker encoder + Tacotron 2 + WaveNet, 3 mustaqil komponent) | ma'lumot topilmadi | "thousands of speakers" (noizli nutq, transkriptsiyasiz) | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | Yo'q | — (vazn chiqarilmagan) | "seconds of reference speech" — aniq raqam oshkor qilinmagan | https://arxiv.org/abs/1806.04558 |
| **VALL-E** | Microsoft | 2023 | **AR codec LM** (+ NAR bosqich) | **60K soat** inglizcha; "hundreds of times larger than existing systems" | ma'lumot topilmadi | 1 (en) | ma'lumot topilmadi | ma'lumot topilmadi | Yo'q | — | **3 sekund** ("only a 3-second enrolled recording") | https://arxiv.org/abs/2301.02111 |
| **VALL-E X** (Microsoft paper) | Microsoft | 2023 | AR codec LM, ko'p tilli | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | Yo'q (kod/vazn chiqarilmagan) | — | ma'lumot topilmadi | — (ochiq reimplementatsiya uchun 1b bo'limiga qarang) |
| **VALL-E 2** | Microsoft | 2024 | AR codec LM + Repetition Aware Sampling + Grouped Code Modeling | ma'lumot topilmadi | ma'lumot topilmadi | 1 (en) | ma'lumot topilmadi | ma'lumot topilmadi | Yo'q | — | ma'lumot topilmadi | https://arxiv.org/abs/2406.05370 |
| **NaturalSpeech 2** | Microsoft | 2023 | latent diffusion + RVQ neyro kodek | **44 000 soat** (nutq + qo'shiq) | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | Yo'q | — | ma'lumot topilmadi | https://arxiv.org/abs/2304.09116 |
| **NaturalSpeech 3** | Microsoft | 2024 | factorized diffusion + FACodec (content/prosody/timbre/acoustic ajratish) | **200 000 soat** | ma'lumot topilmadi | ma'lumot topilmadi | **1B** | ma'lumot topilmadi | Yo'q (faqat FACodec alohida chiqargan) | — | ma'lumot topilmadi | https://arxiv.org/abs/2403.03100 |
| **Voicebox** | Meta | 2023 | **flow matching** (NAR) | **50 000+ soat** "unfiltered, unenhanced speech" | ma'lumot topilmadi | ko'p tilli (aniq son: ma'lumot topilmadi) | ma'lumot topilmadi | ma'lumot topilmadi | Yo'q | — | ma'lumot topilmadi | https://arxiv.org/abs/2306.15687 |
| **Audiobox** | Meta | 2023 | flow matching + Bespoke Solvers | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | Yo'q | — | ma'lumot topilmadi | https://arxiv.org/abs/2312.15821 |
| **SPEAR-TTS** | Google | 2023 | ikki bosqichli token LM (semantic "reading" → acoustic "speaking") | **15 daqiqa parallel (transkriptli) ma'lumot** + katta audio-only korpus | ma'lumot topilmadi | 1 (en) | ma'lumot topilmadi | ma'lumot topilmadi | Yo'q | — | **3 sekund** ("only a short sample of 3 seconds, without any explicit speaker representation") | https://arxiv.org/abs/2302.03540 |
| **SoundStorm** | Google | 2023 | **masked** parallel decoding (bidirectional attention, confidence-based) | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | inference: TPU-v4 (30 s audio 0.5 s'da) | Yo'q | — | ma'lumot topilmadi | https://arxiv.org/abs/2305.09636 |
| **E2-TTS** | Microsoft | 2024 | flow matching (to'liq NAR, filler tokens, duration model YO'Q, G2P YO'Q) | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ~**333.2M** (F5-TTS paperidagi reimplementatsiya: "a 333.2M flat U-Net equipped Transformer") | ma'lumot topilmadi | Yo'q (rasmiy vazn); F5-TTS repo'da reimplementatsiya bor | — | ma'lumot topilmadi | https://arxiv.org/abs/2406.18009 |
| **Seed-TTS** | ByteDance (46 muallif) | 2024 | AR codec LM **va** Seed-TTS_DiT (NAR diffusion, duration-free) | oshkor qilinmagan — faqat "orders of magnitudes larger than the previously largest TTS systems" | ma'lumot topilmadi | ma'lumot topilmadi (en + zh baholangan) | ma'lumot topilmadi | ma'lumot topilmadi | Yo'q | — | ICL tajribalarida **3–20 sekund**; speaker fine-tuning solishtirmasida o'rtacha 15 s; bazaviy model baholashda 20 s | https://arxiv.org/abs/2406.02430 |
| **MegaTTS 3 / S-DiT** | Zhejiang University + **ByteDance** + Inner Mongolia University | 2025 | **latent diffusion transformer** + sparse alignment | **LibriLight 60 000 soat**; scaling tajribalarida ichki ko'p tilli korpus **600 000 soat** | ma'lumot topilmadi | 2 (en, zh) | **0.3B (339M)**; scaling 0.5B–7.0B | **8× A100 80GB** | Ha, lekin cheklangan (pastga qarang) | Apache-2.0, **ammo**: "For security issues, we do not upload the parameters of WaveVAE encoder" — ixtiyoriy ovoz klonlash amalda bloklangan | ma'lumot topilmadi | https://arxiv.org/abs/2502.18924 · https://huggingface.co/ByteDance/MegaTTS3 |
| **TTS-1 / TTS-1-Max** | Inworld | 2025 | AR Transformer SpeechLM (pre-train → SFT → RL-alignment) | ma'lumot topilmadi | ma'lumot topilmadi | **11** | TTS-1: **1.6B**; TTS-1-Max: **8.8B** | ma'lumot topilmadi | vazn holati: ma'lumot topilmadi | Trening/modellashtirish **KODI MIT**; vazn litsenziyasi: ma'lumot topilmadi | "purely in-context learning of the speaker's voice" — uzunlik: ma'lumot topilmadi | https://arxiv.org/abs/2507.21138 |

### 1b. Ochiq vaznli modellar

| Model | Tashkilot | Yil | Arxitektura oilasi | Trening soati | Spiker soni | Til soni | Parametr | Trening GPU | Ochiq vazn | Litsenziya (tijorat?) | Referens audio | Link |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Real-Time-Voice-Cloning** | CorentinJ (shaxsiy loyiha) | 2019 | speaker-embedding (SV2TTS reimplementatsiyasi) | ma'lumot topilmadi | ma'lumot topilmadi | 1 (en) | ma'lumot topilmadi | ma'lumot topilmadi | **Ha** | **ma'lumot topilmadi** — repo'da LICENSE fayl bor, lekin matni olinmadi (404) | **5 sekund** ("Clone a voice in 5 seconds to generate arbitrary speech in real-time") | https://github.com/CorentinJ/Real-Time-Voice-Cloning |
| **YourTTS** | Coqui | 2021 | VITS asosida + speaker embedding (flow-based) | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | **Ha** (Coqui TTS ichida) | Coqui TTS **kodi MPL-2.0**; YourTTS vazni uchun aniq litsenziya: ma'lumot topilmadi | zero-shot; fine-tune uchun "less than 1 minute of speech" | https://arxiv.org/abs/2112.02418 |
| **Tortoise-TTS** | James Betker (shaxsiy) | 2022–2023 | AR transformer + DDPM diffusion dekoder + CLVP reranker | ma'lumot topilmadi | ma'lumot topilmadi | 1 (en) | ma'lumot topilmadi | ma'lumot topilmadi | **Ha** | **Apache-2.0 — tijoratga RUXSAT** | bir nechta referens klip (aniq uzunlik: ma'lumot topilmadi) | https://github.com/neonbjb/tortoise-tts · https://arxiv.org/abs/2305.07243 |
| **VALL-E X** (ochiq reimplementatsiya) | Plachtaa (jamoa) | 2023 | AR codec LM | ma'lumot topilmadi | ma'lumot topilmadi | 3 (en, ja, zh) | ma'lumot topilmadi | ma'lumot topilmadi | **Ha** | **MIT — tijoratga RUXSAT** | **3–10 sekund** | https://github.com/Plachtaa/VALL-E-X |
| **XTTS v1 / v2** | Coqui | 2023 | AR (GPT-uslubidagi) + speaker latent + HiFi-GAN dekoder | ma'lumot topilmadi | ma'lumot topilmadi | **v2: 17** (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko, hi) | ma'lumot topilmadi | ma'lumot topilmadi | **Ha** | ⚠️ **Coqui Public Model License 1.0.0 — TIJORAT TAQIQLANGAN**: "for any non-commercial purpose"; model yoki uning chiqishidan "any direct or indirect payment" olish mumkin emas; tijorat modellarini o'qitishda ishlatish ham taqiqlangan | **6 sekund** ("6-second audio clip") | https://huggingface.co/coqui/XTTS-v2 · LICENSE.txt |
| **GPT-SoVITS** | RVC-Boss (jamoa) | 2024 | AR GPT (semantik tokenlar) + SoVITS (VITS) dekoder | ma'lumot topilmadi | ma'lumot topilmadi | 5 (en, ja, ko, yue, zh) | ma'lumot topilmadi | ma'lumot topilmadi | **Ha** | **MIT — tijoratga RUXSAT** | zero-shot: **5 sekund**; few-shot fine-tune: **1 daqiqa** | https://github.com/RVC-Boss/GPT-SoVITS |
| **OpenVoice v1 / v2** | MyShell.ai | 2024 | tone-color converter (bazaviy TTS + timbre konvertori) | ma'lumot topilmadi | ma'lumot topilmadi | V2: 6 natively (en, es, fr, zh, ja, ko); referens istalgan tilda | ma'lumot topilmadi | ma'lumot topilmadi | **Ha** | **MIT** — "Starting from April 2024, both V2 and V1 are released under MIT License. Free for commercial use." | ma'lumot topilmadi | https://huggingface.co/myshell-ai/OpenVoiceV2 |
| **CosyVoice (1)** | Alibaba (Tongyi / FunAudioLLM) | 2024 | LLM (supervised semantic tokens) + conditional flow matching | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | CosyVoice-300M → **300M** | ma'lumot topilmadi | **Ha** | **Apache-2.0 — tijoratga RUXSAT** | ma'lumot topilmadi | https://arxiv.org/abs/2407.05407 · https://github.com/FunAudioLLM/CosyVoice |
| **CosyVoice 2** | Alibaba | 2024 | LLM (FSQ tokenizer) + **chunk-aware causal flow matching** (streaming) | **zh 130 000 + en 30 000 + ja 4 600 + ko 2 200 soat** (jami ≈166 800); tokenizer o'qitish uchun **200 000 soat** (zh 110 884 + en 99 918) | ma'lumot topilmadi | **9 til** (zh, en, ja, ko, de, es, fr, it, ru) + **18+ xitoy dialekti** | **0.5B** | ma'lumot topilmadi | **Ha** | **Apache-2.0 — tijoratga RUXSAT** | ma'lumot topilmadi | https://arxiv.org/abs/2412.10117 · https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B |
| **CosyVoice 3** | Alibaba | 2025 | LLM + chunk-aware flow matching + differentiable reward model (post-training) | **1 000 000 soat** ⚠️ paper "expanded from 10,000 hours to 1 million hours" deydi — bu CosyVoice 2 paperidagi ≈166 800 soatlik jadval bilan **zid**; manbalar farq qiladi | ma'lumot topilmadi | **9 til + 18 xitoy dialekti** | **1.5B** (0.5B → 1.5B) | ma'lumot topilmadi | **Ha** (repo'da `Fun-CosyVoice3-0.5B-2512`) | **Apache-2.0 — tijoratga RUXSAT** | ma'lumot topilmadi | https://arxiv.org/abs/2505.17589 |
| **FireRedTTS** | FireRedTeam | 2024 | LLM + acoustic LLM dekoder / flow matching dekoder | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi (zh namunalari) | ma'lumot topilmadi | ma'lumot topilmadi | **Ha** | **MPL-2.0 — tijoratga ruxsat** (fayl-darajali copyleft shartlari bilan) | **3–10 sekund** ("The reference audio should not be too long or too short; a duration of 3 to 10 seconds is recommended") | https://github.com/FireRedTeam/FireRedTTS |
| **MaskGCT** | Amphion / open-mmlab (mualliflar: Yuancheng Wang … Zhizheng Wu) | 2024 | **masked generative** codec transformer (NAR, 2 bosqich, alignment YO'Q, duration prediction YO'Q) | **100 000 soat** Emilia ("English and Chinese data from Emilia, each with 50K hours of speech (totaling 100K hours)") | ma'lumot topilmadi | ⚠️ HF kartasi 6 til tegi beradi (en, zh, ko, ja, fr, de), lekin trening bo'limi faqat **en + zh** deydi — **manbalar zid** | T2S-Base **315M**, T2S-Large **695M**, S2A **353M** (jami: ma'lumot topilmadi) | **8× A100 80GB** | **Ha** | ⚠️ **CC-BY-NC-4.0 — TIJORAT TAQIQLANGAN** | ma'lumot topilmadi | https://huggingface.co/amphion/MaskGCT · https://arxiv.org/abs/2409.00750 |
| **F5-TTS** | mualliflar: Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, Xie Chen (tashkilot nomi arXiv sahifasida ko'rsatilmagan) | 2024 | **flow matching** (DiT, to'liq NAR, ConvNeXt text refinement) | ⚠️ paperning abstrakti "a public 100K hours multilingual dataset" deydi, trening bo'limi esa filtrlangan Emilia'dan "**approximately 95K hours** of English and Chinese data" deydi — **manbalar zid** | ma'lumot topilmadi | 2 (en, zh) | **335.8M** | **8× NVIDIA A100 80G**, "for over one week", **1.2M update**, batch 307 200 audio frame (0.91 soat) | **Ha** | ⚠️ **Kod MIT, LEKIN vaznlar CC-BY-NC** — "Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia". **Tijorat TAQIQLANGAN.** Bu eng ko'p xato xabar qilinadigan litsenziya | ma'lumot topilmadi | https://github.com/SWivid/F5-TTS · https://arxiv.org/abs/2410.06885 |
| **Fish-Speech 1.5** | Fish Audio | 2024 | DualAR (ikki darajali AR) | **1 000 000+ soat** | ma'lumot topilmadi | **13** (en, zh, ja, de, fr, es, ko, ar, ru, nl, it, pl, pt) | ma'lumot topilmadi | ma'lumot topilmadi | **Ha** | ⚠️ **CC-BY-NC-SA-4.0 — TIJORAT TAQIQLANGAN** | ma'lumot topilmadi | https://huggingface.co/fishaudio/fish-speech-1.5 |
| **OpenAudio S1-mini** | Fish Audio | 2025 | DualAR (distillangan) | **2 000 000+ soat** | ma'lumot topilmadi | **13** | **0.5B** | ma'lumot topilmadi | **Ha** | ⚠️ **CC-BY-NC-SA-4.0 — TIJORAT TAQIQLANGAN** | ma'lumot topilmadi | https://huggingface.co/fishaudio/openaudio-s1-mini |
| **OpenAudio S2 Pro** | Fish Audio | 2026 | **Dual-AR**: Slow AR 4B (vaqt o'qi) + Fast AR 400M (residual kodbooklar) | **10 000 000+ soat** | ma'lumot topilmadi | **80+ til** (fonema yoki tilga xos preprocessing talab qilmaydi) | **4B + 400M** | ma'lumot topilmadi | **Ha** | ⚠️ **"FISH AUDIO RESEARCH LICENSE"** (kod va vazn) — tijorat holati: nomidan tadqiqotga cheklangan ko'rinadi, aniq shartlar: ma'lumot topilmadi | **10–30 sekund** ("short reference samples (typically 10-30 seconds)") | https://github.com/fishaudio/fish-speech |
| **Llasa** (1B / 3B / 8B) | HKUST Audio + ASLP@NPU + Univ. of Science and Technology Beijing va boshqalar | 2025 | **AR LM** (Llama-3.2 + XCodec2, 65 536 tokenli kodbook) | **250 000 soat** zh+en; 80k va 160k soatlik variantlar ham sinovdan o'tgan | ma'lumot topilmadi | 2 (zh, en) | 1B / 3B / 8B — ⚠️ Llasa-3B HF kartasi "**4B params**" ko'rsatadi (bazaviy Llama-3.2-3B ustiga kengaytirilgan vocab) — **manbalar zid** | ma'lumot topilmadi | **Ha** | ⚠️ **CC-BY-NC-4.0 — TIJORAT TAQIQLANGAN** | ma'lumot topilmadi | https://huggingface.co/HKUSTAudio/Llasa-3B · https://arxiv.org/abs/2502.04128 |
| **Spark-TTS** | HKUST + Shanghai Mobvoi + SJTU + NTU + ASLP@NPU + NetEase Fuxi AI Lab + SparkAudio jamoasi | 2025 | **AR LM** (Qwen2.5-0.5B) + **BiCodec** single-stream kodek (semantik + fixed-length global tokenlar) | ⚠️ abstrakt "a meticulously curated **100,000-hour** dataset" deydi; paper tanasi "VoxBox comprises 4.7 million audio files, sourced from 29 open datasets, totaling **102.5k hours**" deydi (zh 47 627.15 + en 54 873.11) — kichik zidlik | ma'lumot topilmadi | 2 (zh, en) | **0.5B** | ma'lumot topilmadi | **Ha** | ⚠️ **CC-BY-NC-SA-4.0** — "The model can only be used for non-commercial purposes" | ma'lumot topilmadi | https://huggingface.co/SparkAudio/Spark-TTS-0.5B · https://arxiv.org/abs/2503.01710 |
| **IndexTTS 1 / 1.5** | bilibili | 2025 | AR | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | **Ha** | **bilibili Model Use License Agreement** (pastga qarang) | ma'lumot topilmadi | https://github.com/index-tts/index-tts · https://arxiv.org/abs/2502.05512 |
| **IndexTTS 2** | **Artificial Intelligence Platform Department, bilibili, China** | 2025 | **AR** (GPT latent reprezentatsiyalari, timbre↔emotsiya ajratilgan, aniq davomiylik nazorati) | **55 000 soat** (30K zh + 25K en) + **135 soat emotsional ma'lumot (361 spiker)** | emotsional qism: **361 spiker**; umumiy: ma'lumot topilmadi | HF tegi: en, zh. (IndexTTS-2.5: zh, en, ja, es, ar) | ma'lumot topilmadi (IndexTTS-**2.5**: **0.8B**) | **8× NVIDIA A100 80GB**, ~3 hafta | **Ha** | **bilibili Model Use License Agreement** — **tijoratga RUXSAT**, lekin: >100 mln oylik aktiv foydalanuvchi YOKI >1 mlrd RMB yillik daromadda alohida yozma litsenziya kerak; boshqa AI modellarini yaxshilashda ishlatish taqiqlangan; tibbiy diagnostika / avtonom haydash / harbiy / kritik infratuzilma taqiqlangan | ma'lumot topilmadi | https://huggingface.co/IndexTeam/IndexTTS-2 · https://arxiv.org/abs/2506.21619 · LICENSE |
| **Vevo** | Amphion / open-mmlab | 2025 | AR content-style transformer + **flow-matching** acoustic transformer + Vocos vocoder | ⚠️ paper: "**60 000 soat** audiokitob nutqi"; HF kartasi: "Emilia … containing **101k hours** of speech data among six languages" — **manbalar zid** (ehtimol turli chekpointlar) | ma'lumot topilmadi | 6 (en, zh, de, fr, ja, ko) | AR transformer **480M–740M** + FM transformer **330M** + vocoder **250M** | ma'lumot topilmadi | **Ha** | ⚠️ **CC-BY-NC-4.0 — TIJORAT TAQIQLANGAN** | ma'lumot topilmadi | https://huggingface.co/amphion/Vevo · https://arxiv.org/abs/2502.07243 |
| **VibeVoice** (TTS-1.5B / Realtime-0.5B / ASR-7B) | Microsoft | 2025 | **continuous speech tokenizer + next-token diffusion** (LLM = Qwen2.5-1.5B, diffusion dekodlash boshi); akustik tokenizer 24 kHz'dan 3200× downsampling, Encodec'dan 80× yaxshi siqish | ma'lumot topilmadi | **bir dialogda 4 ta spikergacha** | "English and Chinese" (+ Realtime-0.5B kartasida 9 qo'shimcha til "exploration" uchun: de, fr, it, ja, ko, nl, pl, pt, es) | ⚠️ VibeVoice-1.5B HF kartasi "**3B params**" ko'rsatadi (nom 1.5B) — **manbalar zid**. Realtime: **0.5B** (dekoder ~340M + diffusion head ~40M). ASR: **7B**. (VibeVoice-**Large** kartasi olinmadi — HTTP 401) | ma'lumot topilmadi ("curriculum increasing up to 65 536 tokens") | **Ha** (vaznlar HF'da) | ⚠️ **MIT**, LEKIN model kartasi: "limited to research purpose use"; "**We do not recommend using VibeVoice in commercial or real-world applications** without further testing and development". Microsoft TTS **kodini repodan olib tashladi**: "we have removed the VibeVoice-TTS code from this repository" | ma'lumot topilmadi | https://huggingface.co/microsoft/VibeVoice-1.5B · https://github.com/microsoft/VibeVoice · https://arxiv.org/abs/2508.19205 |
| **Higgs Audio v2** | Boson AI | 2025 | **AR codec LM** (Llama-3.2-3B + **DualFFN** audio adapter, unified audio tokenizer @ 25 fps, non-diffusion enkoder/dekoder) | **10 000 000+ soat** ("AudioVerse", avtomatik annotatsiya quvuri) | ma'lumot topilmadi | **4** (en, zh, de, ko) | **5.8B** (3.6B LLM + 2.2B DualFFN) | ma'lumot topilmadi | **Ha** | ⚠️ HF kartasida `license: other`; LICENSE fayl = **"Boson Higgs Audio 2 Community License Agreement"** — tijoratga ruxsat, **lekin >100 000 yillik aktiv foydalanuvchida** Boson AI'dan kengaytirilgan litsenziya kerak. **Ko'p ikkilamchi manba "Apache-2.0" deydi — bu LICENSE fayliga ZID va tasdiqlanmadi.** | ma'lumot topilmadi | https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base · LICENSE |
| **Higgs Audio v3** | Boson AI | 2026 | AR codec LM (v2 davomi) | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | ma'lumot topilmadi | Ha (vaznlar) | ⚠️ **"Boson Higgs Audio v3 Research and Non-Commercial License"** — "Production / hosted / revenue-generating use requires a separate commercial license" | ma'lumot topilmadi | https://github.com/boson-ai/higgs-audio |
| **Sesame CSM-1B** | Sesame | 2025 | **AR codec LM** — "a Llama backbone and a smaller audio decoder that produces Mimi audio codes" (RVQ) | ma'lumot topilmadi | ma'lumot topilmadi | asosan en ("some capacity for non-English … due to data contamination … but it likely won't do well") | ⚠️ nom "1B", HF kartasi "**2B params**" ko'rsatadi — **manbalar zid** | ma'lumot topilmadi | **Ha** | **Apache-2.0 — tijoratga RUXSAT** | ma'lumot topilmadi | https://huggingface.co/sesame/csm-1b · https://github.com/SesameAILabs/csm |
| **Marvis-TTS 250M** | Marvis-AI (Prince Canuma, Lucas Newman) | 2025 (26-avg) | CSM-1B asosida, RVQ tokenlar ustida ishlaydi | Emilia-YODAS (pretrain) + "Expressive Speech" (post-train); soat: ma'lumot topilmadi | ma'lumot topilmadi | en (de, pt, fr, zh "coming soon") | **250M backbone + 60M audio dekoder** (jami 0.6B) | ma'lumot topilmadi | **Ha** | **Apache-2.0 — tijoratga RUXSAT** | **10 sekund** ("Voice cloning quality is dependent on the clarity and quality of the 10-second reference audio") | https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.1 |
| **Orpheus-3B** | Canopy Labs | 2025 | AR LM (Llama-3.2-3B-Instruct + SNAC kodek) | ma'lumot topilmadi | ma'lumot topilmadi | 1 (en) | ⚠️ nom "3B", HF kartasi "**4B params**" — **manbalar zid** | ma'lumot topilmadi | **Ha** | **Apache-2.0 — tijoratga RUXSAT** | ma'lumot topilmadi (karta: "Zero-Shot Voice Cloning: Clone voices without prior fine-tuning") | https://huggingface.co/canopylabs/orpheus-3b-0.1-ft |
| **Zonos-v0.1** (transformer / hybrid) | Zyphra | 2025 | AR — eSpeak fonemizatsiya → DAC token prediction, transformer yoki hybrid (Mamba) backbone | **200 000+ soat** ko'p tilli nutq | ma'lumot topilmadi | 5 (en, ja, zh, fr, de) | **2B** (HF) | ma'lumot topilmadi | **Ha** | **Apache-2.0 — tijoratga RUXSAT** | ⚠️ **"10-30s speaker sample"**, lekin boshqa joyda "a reference clip spanning just a few seconds" — **repo ichida zidlik** | https://huggingface.co/Zyphra/Zonos-v0.1-transformer · https://github.com/Zyphra/Zonos |
| **Chatterbox** (+ Multilingual / Turbo / Nano) | Resemble AI | 2025 | **AR codec LM** — "0.5B Llama backbone"; Turbo: speech-token→mel dekoder 10 qadamdan 1 qadamga distillangan | **0.5M soat** ("Trained on 0.5M hours of cleaned data") | ma'lumot topilmadi | **23+** (ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh) | **0.5B**; Turbo **350M**; Nano **110M** | ma'lumot topilmadi | **Ha** | **MIT — tijoratga RUXSAT**. Eslatma: har bir chiqishga Resemble'ning **Perth** neyro suv belgisi qo'yiladi ("nearly 100% detection accuracy") | ma'lumot topilmadi — repo faqat "a reference clip for voice cloning" deydi | https://huggingface.co/ResembleAI/chatterbox · https://github.com/resemble-ai/chatterbox |
| **Dia-1.6B** | Nari Labs | 2025 | AR (SoundStorm / Parakeet / Descript Audio Codec ta'sirida), dialog generatsiyasi | ma'lumot topilmadi | dialog (spiker teglari bilan) | 1 (en) — "The model only supports English generation at the moment" | **1.6B** | ma'lumot topilmadi | **Ha** | **Apache-2.0 — tijoratga RUXSAT** | ma'lumot topilmadi (`example/voice_clone.py` mavjud) | https://huggingface.co/nari-labs/Dia-1.6B |
| **Supertonic** | Supertone Inc. | 2025 | on-device ONNX Runtime; **zero-shot klonlash haqida hujjatda ma'lumot YO'Q** — preset ovoz stillari (masalan "M1") | ma'lumot topilmadi | preset ovozlar | ma'lumot topilmadi | **66M** | ma'lumot topilmadi | **Ha** | ⚠️ **Namuna kodi MIT; MODELNING O'ZI OpenRAIL-M** (foydalanish cheklovlari bilan) | zero-shot klonlash qo'llab-quvvatlanishi tasdiqlanmadi | https://huggingface.co/Supertone/supertonic |

### 1c. Zero-shot klonlash QILMAYDIGAN, lekin shu ro'yxatda tez-tez adashib keltiriladigan modellar

| Model | Tashkilot | Yil | Arxitektura | Trening soati | Spiker | Til | Parametr | Ochiq vazn | Litsenziya | Referens audio | Link |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Kokoro-82M** | hexgrad | 2025 | StyleTTS 2 + ISTFTNet dekoder | "A few hundred hours of audio" | **54 preset ovoz** (v1.0) | 8 | **82M** | **Ha** | **Apache-2.0** | ❌ **ZERO-SHOT KLONLASH YO'Q** — faqat preset ovozlar (`af_heart` kabi). Model kartasida referens-audio klonlash haqida bir og'iz so'z yo'q | https://huggingface.co/hexgrad/Kokoro-82M |
| **Parler-TTS Large v1** | Hugging Face | 2024 | T5-uslubidagi enkoder-dekoder + DAC; **matnli ta'rif** bilan boshqarish | **45 000 soat** (LibriTTS-R + MLS English) | **34 nomlangan spiker** | 1 (en) | **2.2B** | **Ha** | **Apache-2.0** | ❌ referens audio ishlatmaydi — faqat matnli prompt ("gender, background noise, speaking rate, pitch, reverberation") | https://huggingface.co/parler-tts/parler-tts-large-v1 |
| **Maya1** | Maya Research | 2025 | Llama-uslubidagi 3B transformer + SNAC kodek | "Internet-scale English speech corpus" + proprietar studiya yozuvlari (soat: ma'lumot topilmadi) | ma'lumot topilmadi | en (multi-accent) | **3B** | **Ha** | **Apache-2.0** | ❌ **ta'rif asosida ovoz dizayni, klonlash emas**: `<description="40-year-old, warm, low pitch, conversational">`. "No training data. Just describe and generate." | https://huggingface.co/maya-research/maya1 |

### 1d. Tijorat / yopiq tizimlar

Bu qatorlarda arxitektura va ma'lumot ustunlari deyarli hamma joyda **oshkor qilinmagan** — bu faktning o'zi muhim xulosa.

| Tizim | Tashkilot | Yil | Arxitektura (oshkor qilingani) | Trening ma'lumoti | Til | Parametr | Ochiq vazn | Litsenziya | **Klonlash uchun minimal referens audio** | Link |
|---|---|---|---|---|---|---|---|---|---|---|
| **ElevenLabs** (v2/v3/v4) | ElevenLabs | 2023– | **oshkor qilinmagan** | **oshkor qilinmagan** | ma'lumot topilmadi | oshkor qilinmagan | Yo'q | Tijorat SaaS | **Instant Voice Cloning: "1–2 minutes of good audio"**; Professional Voice Cloning: **"30–180 minutes of good audio"**. Eslatma: "Having more than 2-3 minutes of audio will yield little improvement and can, in some cases, even be detrimental to the stability of the clone" | https://elevenlabs.io/docs/product-guides/voices/voice-cloning |
| **PlayHT / Play AI** (PlayDialog, Play 3.0 mini) | PlayHT | — | **oshkor qilinmagan** | oshkor qilinmagan | ma'lumot topilmadi | oshkor qilinmagan | Yo'q | Tijorat SaaS | **30 sekund** ("30 seconds of speech" instant klon uchun) | https://docs.play.ht/reference/api-getting-started |
| **Resemble AI** | Resemble AI | — | **oshkor qilinmagan** (ochiq modeli — Chatterbox, yuqoriga qarang) | oshkor qilinmagan | ma'lumot topilmadi | oshkor qilinmagan | Yo'q (lekin Chatterbox MIT ostida ochiq) | Tijorat SaaS | **10 sekund** ("Create natural-sounding voice clones with just 10 seconds of audio"); Professional clone: Business plan+, trening <1 daqiqa | https://docs.resemble.ai/voice-creation/voices/clone-overview |
| **Cartesia Sonic** (3.5 / 3.6) | Cartesia | 2024– | **oshkor qilinmagan** (SSM/Mamba asosi keng aytiladi, rasmiy tasdiq topilmadi) | oshkor qilinmagan | ma'lumot topilmadi | oshkor qilinmagan | Yo'q | Tijorat SaaS | **10 sekund** ("10 seconds of audio is enough to get started, and up to 60 seconds to better retain the speaker's accent"). "The 60-second limit applies to Sonic 3.6 and newer. Older models only learn from the first 10 seconds" | https://docs.cartesia.ai/build-with-cartesia/capability-guides/clone-voices |
| **MiniMax Speech-02 / MiniMax-Speech** | MiniMax | 2025 | AR Transformer + **learnable speaker encoder** — "extracts timbre features from a reference audio **without requiring its transcription**" (bu qism oshkor qilingan; qolgani emas) | oshkor qilinmagan | **32 til** | oshkor qilinmagan | Yo'q | Tijorat API | ⚠️ **manbalar zid**: rasmiy hujjat **10 sekund – 5 daqiqa**; fal.ai endpointi **≥10 sekund**; Replicate hujjati **5 sekund** deydi | https://arxiv.org/abs/2505.07916 |
| **Hume Octave** | Hume AI | 2025 | "Octave, Hume's speech-language model, deeply models language and speech patterns" — LLM asosida; batafsil **oshkor qilinmagan** | oshkor qilinmagan | ma'lumot topilmadi | oshkor qilinmagan | Yo'q | Tijorat API | ⚠️ **manbalar zid**: Hume blogi **"recordings as brief as 5 seconds"**; ba'zi 2026 ikkilamchi manbalar 15 sekund deydi. Rasmiy dev-hujjatda minimal uzunlik topilmadi | https://www.hume.ai/blog/how-to-clone-your-voice-with-ai |
| **OpenAI Voice Engine** | OpenAI | 2024 (preview) | **oshkor qilinmagan** | oshkor qilinmagan | ma'lumot topilmadi | oshkor qilinmagan | Yo'q | **Umumiy chiqarilmagan** — faqat ~10 ishlab chiquvchiga kichik preview | **"a single 15-second audio sample"** | https://community.openai.com/t/voice-engine-navigating-the-challenges-and-opportunities-of-synthetic-voices/701714 |
| **Google Chirp 3: HD** | Google | 2025 | "Powered by our latest generation of generative models" — boshqa hech narsa **oshkor qilinmagan** | oshkor qilinmagan | 60+ BCP-47 til kodi, 28 ovoz nomi | oshkor qilinmagan | Yo'q | Tijorat (Google Cloud) | HD ovozlar klonlash emas — preset | https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd |
| **Google Chirp 3: Instant Custom Voice** | Google | 2025 | **oshkor qilinmagan** | oshkor qilinmagan | 30+ til | oshkor qilinmagan | Yo'q | Tijorat, **allowlist**: "Access to Instant Custom Voice is restricted to allow-listed users" | **10 sekundgacha** ("record up to 10 seconds of audio") **+ majburiy 10 sekundlik rozilik yozuvi** Google skripti bo'yicha: "I am the owner of this voice and I consent to Google using this voice to create a synthetic voice model" | https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice |
| **Gemini TTS** (3.1 / 3.8 Flash TTS) | Google | 2025–2026 | **oshkor qilinmagan** | oshkor qilinmagan | ma'lumot topilmadi | oshkor qilinmagan | Yo'q | Tijorat API | ma'lumot topilmadi | https://artificialanalysis.ai/text-to-speech/leaderboard |

---

## 2. HAQIQATAN OCHIQ **VA** TIJORATGA RUXSAT ETILGAN modellar

Bu qisqa ro'yxat — eng qimmatli ma'lumot. Faqat Apache-2.0 / MIT, cheklovsiz.

| Model | Litsenziya | Klonlash bor? | Til | Diqqat qiling |
|---|---|---|---|---|
| **CosyVoice 1 / 2 / 3** | **Apache-2.0** | Ha | 9 til + 18 xitoy dialekti | Eng toza huquqiy holat + eng ko'p til. Solo dev uchun №1 |
| **Chatterbox** (+ Multilingual / Turbo / Nano) | **MIT** | Ha | 23+ | Har bir chiqishda Perth suv belgisi; bu majburiy |
| **GPT-SoVITS** | **MIT** | Ha (5 s zero-shot) | 5 (en, ja, ko, yue, zh) | 1 daqiqa bilan fine-tune — past resursli til uchun eng amaliy ko'prik |
| **OpenVoice v1 / v2** | **MIT** | Ha (timbre konvertori) | 6 natively, referens istalgan tilda | "Free for commercial use" — rasmiy formulirovka |
| **Zonos-v0.1** | **Apache-2.0** | Ha | 5 | 200k+ soatda o'qitilgan, 2B |
| **Sesame CSM-1B** | **Apache-2.0** | Ha (bazaviy model, aniq ovozga fine-tune qilinmagan) | asosan en | Dialog/suhbat uchun |
| **Orpheus-3B** | **Apache-2.0** | Ha (kartada aniq aytilgan) | en | Llama-3.2 ustiga, SNAC |
| **Marvis-TTS 250M** | **Apache-2.0** | Ha (10 s) | en | 0.6B jami — eng yengil klonlovchi |
| **Dia-1.6B** | **Apache-2.0** | Ha (`voice_clone.py`) | en | Dialog generatsiyasi |
| **Tortoise-TTS** | **Apache-2.0** | Ha | en | Sifatli, lekin eski; 2026-da raqobatchi emas |
| **VALL-E X** (Plachtaa) | **MIT** | Ha (3–10 s) | 3 | Tarixiy qiymat; sifat eskirgan |
| **FireRedTTS** | **MPL-2.0** | Ha (3–10 s) | ma'lumot topilmadi | Copyleft — o'zgartirilgan fayllarni ochiq qoldirish shartlari bor |
| *Kokoro-82M* | Apache-2.0 | ❌ **YO'Q** | 8 | Klonlash kerak bo'lsa — mos emas |
| *Parler-TTS* | Apache-2.0 | ❌ **YO'Q** | en | Matnli ta'rif bilan boshqarish |
| *Maya1* | Apache-2.0 | ❌ **YO'Q** | en | Ta'rif asosida ovoz dizayni |

### ⚠️ Shartli / cheklangan ruxsat (tijorat mumkin, LEKIN shart bilan)

| Model | Litsenziya | Shart |
|---|---|---|
| **IndexTTS 2** | bilibili Model Use License Agreement | >100 mln MAU yoki >1 mlrd RMB daromadda alohida litsenziya kerak; boshqa modellarni o'qitish taqiqlangan; yuqori xavfli sohalar taqiqlangan |
| **Higgs Audio v2** | Boson Higgs Audio 2 Community License | >100 000 yillik aktiv foydalanuvchida kengaytirilgan litsenziya kerak |
| **VibeVoice** | MIT litsenziya, lekin model kartasi "research purpose only" deydi va tijoratni **tavsiya qilmaydi**; Microsoft TTS kodini repodan olib tashlagan | Huquqiy va amaliy noaniqlik |
| **MegaTTS 3** | Apache-2.0, lekin **WaveVAE enkoder vazni berilmagan** | Ixtiyoriy ovozni klonlash amalda ishlamaydi — faqat ByteDance bergan oldindan ekstraktlangan latentlar |
| **Supertonic** | Kod MIT, **model OpenRAIL-M** | OpenRAIL foydalanish cheklovlari qo'llanadi |

### 🚫 CC-BY-NC va maxsus cheklangan (TIJORAT TAQIQLANGAN)

| Model | Aniq litsenziya | Izoh |
|---|---|---|
| **F5-TTS** | **Kod MIT, VAZNLAR CC-BY-NC** | ⚠️ **Eng ko'p noto'g'ri xabar qilinadigan holat.** Repo so'zma-so'z: "Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia". Ko'p blog "F5-TTS MIT" deb yozadi — bu XATO |
| **MaskGCT** | CC-BY-NC-4.0 | |
| **Llasa** (1B/3B/8B) | CC-BY-NC-4.0 | |
| **Vevo** | CC-BY-NC-4.0 | |
| **Spark-TTS** | CC-BY-NC-SA-4.0 — "The model can only be used for non-commercial purposes" | |
| **Fish-Speech 1.5**, **OpenAudio S1-mini** | CC-BY-NC-SA-4.0 | |
| **OpenAudio S2 Pro** | "FISH AUDIO RESEARCH LICENSE" (bespoke) | Aniq tijorat shartlari: ma'lumot topilmadi |
| **XTTS v1 / v2** | **Coqui Public Model License 1.0.0** (bespoke) | So'zma-so'z: "for any non-commercial purpose"; "any direct or indirect payment arising from the use of the model or its output" taqiqlangan; tijorat modellarini o'qitish ham taqiqlangan. `coqui.ai/cpml` sayti o'lgan (404) — matn HF'dagi `LICENSE.txt`da |
| **Higgs Audio v3** | Boson Higgs Audio v3 Research and Non-Commercial License | "Production / hosted / revenue-generating use requires a separate commercial license" |

**Muhim naqsh:** Emilia datasetida o'qitilgan hamma narsa (F5-TTS, MaskGCT, Vevo, Marvis pretrain) CC-BY-NC bo'lib qoladi, chunki Emilia "in-the-wild" (internetdan yig'ilgan) dataset. Apache-2.0 modellar (CosyVoice, Chatterbox) o'z ma'lumotini litsenziyalagan yoki o'zi yig'gan.

---

## 3. BENCHMARK SOLISHTIRISHLARI (haqiqiy raqamlar)

### 3a. Seed-TTS-eval (sanoat standarti) — CosyVoice 2 paperidagi jadval (Table 6)

| Model | test-zh CER (%) ↓ | test-en WER (%) ↓ | test-hard WER (%) ↓ |
|---|---|---|---|
| **Seed-TTS** (yopiq) | **1.12** | **2.25** | 7.59 |
| CosyVoice 2 | 1.45 | 2.57 | **6.83** |
| F5-TTS | 1.56 | **1.83** | 8.67 |
| MaskGCT | 2.27 | 2.62 | 10.27 |
| FireRedTTS | 1.51 | 3.82 | 17.45 |

**Yetakchi:** test-zh va test-en'da Seed-TTS (ByteDance, yopiq); test-hard (qiyin holatlar)da CosyVoice 2; sof inglizcha WER'da F5-TTS.

### 3b. Seed-TTS-eval + SIM — Spark-TTS paperidagi jadval (Table 4)

| Model | test-zh CER ↓ | test-zh SIM ↑ | test-en WER ↓ | test-en SIM ↑ |
|---|---|---|---|---|
| Seed-TTS † (yopiq) | 1.12 | **0.796** | 2.25 | **0.762** |
| Spark-TTS | 1.20 | 0.672 | 1.98 | 0.584 |
| CosyVoice 2 | 1.45 | 0.748 | 2.57 | 0.652 |
| F5-TTS | 1.56 | 0.741 | **1.83** | 0.647 |
| Llasa-8B-250k | 1.59 | 0.684 | 2.97 | 0.574 |

† = yopiq model. **Spiker o'xshashligida (SIM) yopiq Seed-TTS hamon ochiq modellardan sezilarli ustun** (0.796/0.762 vs ~0.75/0.65).

### 3c. Seed-TTS paperining o'z jadvali (Table 1) — inson bilan solishtirish

| Tizim | Til | WER ↓ | SIM ↑ |
|---|---|---|---|
| Seed-TTS | EN | 2.249 | **0.762** |
| **Inson (Human)** | EN | **2.143** | 0.730 |
| **Seed-TTS** | ZH | **1.115** | **0.796** |
| Inson (Human) | ZH | 1.254 | 0.750 |

Xitoy tilida Seed-TTS **insondan** yaxshiroq WER va SIM ko'rsatdi — "inson pariteti" da'volarining asosi.

### 3d. LibriSpeech test-clean — MaskGCT paperidagi jadval

| Tizim | SIM-O ↑ | WER ↓ | CMOS |
|---|---|---|---|
| Ground Truth | 0.68 | 1.94 | 0.00 |
| VALL-E | 0.50 | 5.90 | −0.52±0.22 |
| Voicebox | 0.64 | 2.03 | −0.41±0.13 |
| NaturalSpeech 3 | 0.67 | **1.94** | 0.16±0.14 |
| MaskGCT | **0.687** | 2.634 | 0.10±0.16 |
| MaskGCT (gt length) | **0.697** | 2.011 | 0.13±0.13 |

### 3e. LibriSpeech test-clean — MegaTTS 3 / S-DiT paperidagi jadval

| Metrika | S-DiT | S-DiT-accel. | F5-TTS | CosyVoice | MaskGCT |
|---|---|---|---|---|---|
| WER ↓ | **1.82%** | 1.86% | 1.96% | 2.24% | 2.63% |
| SIM-O ↑ | **0.71** | 0.70 | 0.66 | 0.62 | 0.69 |
| CMOS | **0.00** | −0.03 | −0.12 | −0.18 | — |

### 3f. F5-TTS paperining o'z jadvallari

| Test to'plami | Model | WER (%) ↓ | SIM-o ↑ |
|---|---|---|---|
| LibriSpeech-PC test-clean | F5-TTS (16 NFE) | 2.53 | 0.66 |
| LibriSpeech-PC test-clean | F5-TTS (32 NFE) | 2.42 | 0.66 |
| LibriSpeech-PC test-clean | E2-TTS (32 NFE) | 2.95 | **0.69** |
| Seed-TTS test-en | F5-TTS (16 NFE) | 1.89 | 0.67 |
| Seed-TTS test-en | F5-TTS (32 NFE) | **1.83** | 0.67 |
| Seed-TTS test-en | E2-TTS (32 NFE) | 2.19 | **0.71** |
| Seed-TTS test-zh | F5-TTS (16 NFE) | 1.74 | 0.75 |
| Seed-TTS test-zh | F5-TTS (32 NFE) | **1.56** | **0.76** |
| Seed-TTS test-zh | E2-TTS (32 NFE) | 1.97 | 0.73 |

### 3g. IndexTTS2 paperidagi jadval (Table 1)

⚠️ **OGOHLANTIRISH:** IndexTTS2'ning "SS" (speaker similarity) raqamlari boshqa paperlardagi "SIM-o" dan **ancha yuqori**, chunki boshqa o'xshashlik modeli ishlatilgan. **Bu raqamlarni yuqoridagi 3b/3d jadvallar bilan solishtirib bo'lmaydi** — faqat shu jadval ichida solishtirish mumkin.

| Test | IndexTTS2 | F5-TTS | CosyVoice2 | MaskGCT | SparkTTS |
|---|---|---|---|---|---|
| SeedTTS test-en SS ↑ / WER ↓ | **0.860** / **1.521%** | 0.803 / 1.937% | 0.794 / 3.277% | 0.824 / 2.530% | 0.755 / 1.543% |
| SeedTTS test-zh SS ↑ / WER ↓ | **0.865** / **1.008%** | 0.844 / 1.514% | 0.846 / 1.451% | 0.807 / 2.447% | 0.683 / 2.636% |
| LibriSpeech test-clean SS ↑ / WER ↓ | **0.870** / **3.115%** | 0.821 / 8.044% | 0.843 / 5.999% | 0.790 / 7.759% | 0.756 / 8.843% |

IndexTTS repo'sidagi yangiroq jadval (Table 2): **IndexTTS-2.5 — o'rtacha WER 6.75, o'rtacha SS 73.18**; raqobatchilar WER 5.94–10.95, SS 64.49–73.18 oralig'ida.

### 3h. Higgs Audio v2 vs ElevenLabs (Boson AI o'z benchmarki)

**Seed-TTS Eval + ESD:**

| Model | Seed-TTS WER ↓ | Seed-TTS SIM ↑ | ESD WER ↓ | ESD SIM ↑ |
|---|---|---|---|---|
| Higgs Audio v2 | 2.44 | **67.70** | 1.78 | **86.13** |
| ElevenLabs Multilingual V2 | **1.43** | 50.00 | **1.66** | 65.87 |
| CosyVoice2 | 2.28 | 65.49 | 2.71 | 80.48 |

**EmergentTTS-Eval (gpt-4o-mini-tts ga qarshi g'alaba foizi):**

| Model | Emotions ↑ | Questions ↑ |
|---|---|---|
| **Higgs Audio v2** | **75.71%** | **55.71%** |
| gpt-4o-audio-preview | 61.64% | 47.85% |
| Qwen 2.5 Omni | 41.60% | 51.78% |

**Diqqat:** ElevenLabs aniqlikda (WER 1.43) yetakchi, lekin **spiker o'xshashligida ancha past (SIM 50.00 vs 67.70)** — ya'ni ElevenLabs tushunarli gapiradi, lekin referens ovozga kamroq o'xshaydi. Bu Boson AI'ning o'z o'lchovi, mustaqil tasdiq emas.

### 3i. OpenAudio S2 Pro (Fish Audio o'z benchmarki)

- Seed-TTS Eval WER (xitoy): **0.54%** — "best overall"
- Seed-TTS Eval WER (ingliz): **0.99%** — "best overall"
- EmergentTTS-Eval g'alaba foizi: **81.88%** — "highest overall"

Bu da'volar mustaqil tasdiqlanmagan; dastlabki raqamlar bo'yicha hozirda e'lon qilingan eng yaxshi Seed-TTS-eval natijalari.

### 3j. Voicebox va Audiobox (Meta) raqamlari

- **Voicebox**: WER **1.9%** vs VALL-E **5.9%**; audio o'xshashlik **0.680** vs VALL-E **0.581**; "up to 20 times faster"
- **Audiobox**: LibriSpeech zero-shot TTS o'xshashlik **0.745**; AudioCaps FAD **0.77**; Bespoke Solvers bilan **25× tezlanish**
- **SoundStorm**: TPU-v4'da 30 sekund audio **0.5 sekundda** — AudioLM'dan "two orders of magnitude faster"

### 3k. Artificial Analysis Speech Arena (Elo) — 2026-10-01 holatiga

Ko'r (blind) foydalanuvchi ovozlari asosida:

| O'rin | Model | Elo | Ochiq vazn |
|---|---|---|---|
| 1 | Eleven v4 | **1320±19** | Yo'q |
| 2 | Sonic 3.6 (Cartesia) | 1273±16 | Yo'q |
| 3 | Gemini 3.8 Flash TTS | 1272±16 | Yo'q |
| 4 | Qwen-Audio-3.0-TTS-Plus | 1259±16 | Yo'q |
| 5 | Realtime TTS-2 | 1247±17 | Yo'q |
| 6 | Gemini 3.8 Flash-Lite TTS | 1240±15 | Yo'q |
| 7 | Simba 3.2 | 1239±13 | Yo'q |
| 8 | Luna TTS | 1226±13 | Yo'q |
| 9 | Realtime TTS-2 Flash | 1209±15 | Yo'q |
| 10 | **Breeze TTS 2** | 1206±16 | **HA** ← eng yuqori ochiq vaznli |
| 11 | StepAudio 2.5 TTS (Aug 2026) | 1206±16 | Yo'q |
| 12 | Gemini 3.1 Flash TTS | 1205±12 | Yo'q |
| 13 | v3 Conversational | 1200±15 | Yo'q |
| 14 | Sonic 3.5 | 1187±12 | Yo'q |
| 15 | Soniox TTS Real-Time v2 | 1175±12 | Yo'q |

**Xulosa:** subyektiv idrokda yopiq modellar hamon yetakchi. Eng yaxshi ochiq vaznli model (Breeze TTS 2, 1206) birinchi o'rindagi Eleven v4 (1320) dan **~114 Elo** pastda. Lekin obyektiv WER/SIM benchmarklarida (3a–3g) ochiq modellar yopiqlarga juda yaqin — ya'ni **tafovut aniqlikda emas, ekspressivlik va "jonlilik"da**.

**Olinmagan manba:** TTS Arena V2 (HuggingFace Space `TTS-AGI/TTS-Arena-V2`) HTTP 401 qaytardi — undan raqam keltirilmagan. Leaderboard raqamlari tez o'zgaradi va manbalar orasida farq qiladi (bitta qidiruv natijasi Gemini 3.1 Flash TTS'ni 1214 bilan birinchi deb ko'rsatgan — bu yuqoridagi to'g'ridan-to'g'ri o'qilgan jadval bilan zid, chunki snapshot vaqtlari boshqa).

---

## 4. TARIXIY BURILISH NUQTALARI — qisqa xronologiya

| Sana | Voqea | Nega muhim |
|---|---|---|
| **2018-iyun** | **SV2TTS** (Google, arXiv 1806.04558) | Zero-shot klonlash **g'oyasi**ni birinchi amaliy ko'rsatdi: alohida o'qitilgan speaker encoder'dan olingan embedding Tacotron 2'ni shartlaydi. Sifat robotik, lekin "sekundlar ichida ko'rilmagan ovoz" tamoyili shu yerda tug'ildi |
| **2019** | **Real-Time-Voice-Cloning** (CorentinJ) | SV2TTS'ni ochiq kodga olib chiqdi va "**5 sekundda ovoz klonlash**" shiorini ommalashtirdi. Bugun muallifning o'zi ogohlantiradi: "this repo has quickly gotten old" va Chatterbox'ga yo'naltiradi |
| **2021-dek** | **YourTTS** (Coqui) | VITS asosida zero-shot + ko'p tilli; fine-tune uchun "<1 daqiqa nutq". Past resursli tillar uchun birinchi jiddiy yo'l |
| **2022–2023** | **Tortoise-TTS** (James Betker) | AR transformer + DDPM diffusion — sifatda sakrash, lekin juda sekin. Apache-2.0 |
| **🔴 2023-yanvar** | **VALL-E** (Microsoft) — **BURILISH NUQTASI #1** | **Zero-shot klonlashni haqiqatan ishlatgan paper.** Uchta narsani birlashtirdi: (1) nutqni neyro kodek **diskret tokenlari** sifatida ifodalash, (2) buni **til modeli** vazifasi deb qarash, (3) **60K soat**ga masshtablash — "hundreds of times larger than existing systems". Natija: **3 sekundlik** promptdan in-context learning bilan klonlash, oldingi SOTA'dan sezilarli ustun. Butun 2023–2026 davridagi AR-codec-LM oilasi shu yerdan chiqdi. Vazn chiqarilmagan |
| 2023-fev | SPEAR-TTS (Google) | Faqat **15 daqiqa** transkriptli ma'lumot bilan ishlashni ko'rsatdi (semantic/acoustic ajratish + backtranslation). 3 sekundlik prompt |
| 2023-apr | NaturalSpeech 2 (Microsoft) | 44K soat, latent diffusion — AR LM'ning prosodiya beqarorligi va so'z tashlab ketishiga javob |
| 2023-may | SoundStorm (Google) | Masked parallel decoding — AR'dan **2 tartib tezroq** |
| **2023-iyun** | **Voicebox** (Meta) | **Flow matching NAR** paradigmasini TTS'ga olib keldi: VALL-E'ga nisbatan WER 1.9% vs 5.9%, o'xshashlik 0.680 vs 0.581, **20× tezroq**. Hozirgi F5/E2/MegaTTS3 oilasining ildizi |
| 2023-sent | **XTTS** (Coqui) | Birinchi keng ishlatilgan **ochiq** ko'p tilli klonlash (17 til, 6 sekundlik klip). ⚠️ Lekin CPML — tijorat taqiq. Bu "ochiq ≠ erkin" muammosini birinchi marta keng ko'rsatdi |
| **2024-yanvar** | **GPT-SoVITS** (RVC-Boss) | **MIT** litsenziya, **5 sekund** zero-shot, **1 daqiqa** fine-tune. Yakka ishlab chiquvchilar uchun haqiqiy amaliy burilish — litsenziya toza, resurs talabi kichik |
| 2024-mart | OpenVoice V2 **MIT** ga o'tdi; **OpenAI Voice Engine** (15 sekund) e'lon qilindi, lekin **chiqarilmadi** | "Juda xavfli" argumenti — yopiq laboratoriyalar ortga tortildi, ochiq jamoa oldinga ketdi |
| 2024-mart | NaturalSpeech 3 (Microsoft) | 200K soat, 1B parametr, FACodec bilan atribut ajratish |
| **2024-iyun** | **Seed-TTS** (ByteDance) + **seed-tts-eval** | Xitoy tilida **insondan yaxshiroq** WER (1.115 vs 1.254) va SIM (0.796 vs 0.750). Muhimi: `seed-tts-eval` **sanoat standart benchmarkiga** aylandi — bugun hamma shu jadvalga qarshi hisobot beradi |
| 2024-iyun | E2-TTS (Microsoft) | "Embarrassingly Easy": duration model YO'Q, G2P YO'Q, monotonic alignment YO'Q — faqat flow matching + filler tokens. Soddalashtirish to'lqinini boshladi |
| 2024-iyul | **CosyVoice** (Alibaba) — **Apache-2.0** | Yirik laboratoriyadan birinchi **tijoratga to'liq ruxsat etilgan** zero-shot klonlash |
| **🔴 2024-okt** | **F5-TTS** — **BURILISH NUQTASI #2 (arzon va ochiq)** | **335.8M parametr, 8× A100 80GB, bir hafta, 1.2M qadam, ochiq Emilia datasetida.** Butun retsept oshkor. Shu paytdan boshlab zero-shot klonlash modelini **universitet laboratoriyasi yoki yakka ishlab chiquvchi qayta o'qitishi mumkin** bo'ldi. ⚠️ Lekin vaznlar CC-BY-NC (Emilia sababli) |
| 2024-okt | MaskGCT (Amphion) | 100K soat, masked generative, 8× A100. LibriSpeech'da inson darajasidagi o'xshashlik (SIM-O 0.687 vs GT 0.68). CC-BY-NC |
| 2024-dek | CosyVoice 2 (Apache-2.0, 9 til, streaming); Fish-Speech 1.5 (1M soat, 13 til) | Ko'p tillilik ochiq ekotizimga keldi |
| 2025-fev–may | Llasa (250K soat), Spark-TTS (VoxBox 102.5k soat), Zonos (Apache, 200k soat), **Sesame CSM** (Apache), **Orpheus** (Apache), **Dia** (Apache), IndexTTS2 (bilibili, 55K soat), MegaTTS3 (ByteDance), MiniMax-Speech (32 til) | Portlash yili. LLM backbone (Llama/Qwen) + neyro kodek standart retseptga aylandi |
| **🔴 2025-may** | **Chatterbox** (Resemble AI) — **BURILISH NUQTASI #3** | **MIT litsenziya + 0.5M soat trening + 0.5B parametr + 23+ til.** Tijorat kompaniyasi haqiqatan erkin litsenziya bilan kuchli model chiqardi va "ElevenLabs'dan ustun" deb da'vo qildi. Perth suv belgisi majburiy — bu mas'uliyatli chiqarish modeli |
| 2025-iyul–avg | Higgs Audio v2 (Boson AI, 10M soat, 5.8B); VibeVoice (Microsoft, 90 daqiqa dialog, 4 spiker) | Masshtab yana sakradi. ⚠️ Microsoft keyinchalik VibeVoice-TTS kodini "noto'g'ri ishlatish" sababli repodan olib tashladi |
| 2026 | OpenAudio S2 Pro (**10M soat, 80+ til**, Fish Audio Research License); Chatterbox Turbo/Nano; CosyVoice 3 (1M soat, 1.5B); Higgs v3 non-commercial'ga o'tdi | **Tendensiya:** masshtab oshdi, lekin **litsenziyalar qattiqlashdi** — Higgs v2 (community) → v3 (non-commercial), Fish (CC-BY-NC) → (bespoke research license). Ochiq va erkin modellar oynasi yopilayotgandek |

### Javoblar: qaysi paper nimani qildi?

- **Zero-shot klonlashni birinchi marta ISHLATTIRGAN:** **VALL-E** (Microsoft, 2023-yanvar). Neyro kodek LM + 60K soat + 3 sekundlik prompt. SV2TTS (2018) g'oyani ko'rsatdi, lekin sifat amaliy emas edi.
- **Uni ARZON va OCHIQ qilgan:** **F5-TTS** (2024-oktyabr) — 335.8M parametr, 8× A100, bir hafta, to'liq ochiq retsept + ochiq dataset. Litsenziya jihatidan esa **Chatterbox** (2025-may, MIT) va **CosyVoice** (2024-iyul, Apache-2.0) — haqiqiy erkinlikni berganlar.
- **Benchmarkni standartlashtirgan:** **Seed-TTS** (ByteDance, 2024-iyun) va uning `seed-tts-eval` to'plami.

---

## 5. Umumiy naqshlar (xulosa kuzatishlari)

1. **To'rtta arxitektura oilasi bor, hammasi hozir yashayapti:**
   - *speaker-embedding*: SV2TTS, RTVC, YourTTS, OpenVoice — eski, yengil, sifat past
   - *AR codec LM*: VALL-E, Seed-TTS, CosyVoice, Llasa, Spark-TTS, Chatterbox, Higgs, Sesame CSM, Orpheus, IndexTTS2, Fish/OpenAudio — eng ko'p tarqalgan, ekspressiv, streaming oson
   - *flow matching (NAR)*: Voicebox, Audiobox, E2-TTS, F5-TTS, MegaTTS3 — tez, barqaror, WER yaxshi
   - *masked generative*: SoundStorm, MaskGCT — parallel, juda tez
2. **Referens audio uzunligi 3 sekunddan 2 daqiqagacha tarqalgan.** Eng qisqa: VALL-E va SPEAR-TTS (3 s), Google ICV (10 s), GPT-SoVITS (5 s). Eng uzun talab — paradoksal ravishda **tijorat** ElevenLabs (1–2 daqiqa IVC). Ya'ni "10 sekund yetadi" da'vosi tadqiqot modellarida tijorat modellaridan ko'proq to'g'ri.
3. **Trening soati 15 daqiqadan 10 million soatgacha** — 7 tartib farq. SPEAR-TTS 15 daqiqa parallel ma'lumot bilan ishlashni isbotladi; OpenAudio S2 Pro 10M soat ishlatadi.
4. **Yopiq modellar hech narsani oshkor qilmaydi.** ElevenLabs, Cartesia, OpenAI, Google, PlayHT, Hume — na arxitektura, na ma'lumot soati, na parametr soni. Faqat MiniMax qisman oshkor qildi (learnable speaker encoder) va paper chiqardi.
5. **8× A100 80GB — de-fakto standart trening konfiguratsiyasi.** F5-TTS, MaskGCT, MegaTTS3 va IndexTTS2 — to'rttasi ham aynan shu.

---

## 6. Barcha manbalar

**Paperlar:** [SV2TTS 1806.04558](https://arxiv.org/abs/1806.04558) · [YourTTS 2112.02418](https://arxiv.org/abs/2112.02418) · [Tortoise 2305.07243](https://arxiv.org/abs/2305.07243) · [VALL-E 2301.02111](https://arxiv.org/abs/2301.02111) · [VALL-E 2 2406.05370](https://arxiv.org/abs/2406.05370) · [SPEAR-TTS 2302.03540](https://arxiv.org/abs/2302.03540) · [SoundStorm 2305.09636](https://arxiv.org/abs/2305.09636) · [NaturalSpeech 2 2304.09116](https://arxiv.org/abs/2304.09116) · [NaturalSpeech 3 2403.03100](https://arxiv.org/abs/2403.03100) · [Voicebox 2306.15687](https://arxiv.org/abs/2306.15687) · [Audiobox 2312.15821](https://arxiv.org/abs/2312.15821) · [E2-TTS 2406.18009](https://arxiv.org/abs/2406.18009) · [F5-TTS 2410.06885](https://arxiv.org/abs/2410.06885) · [MaskGCT 2409.00750](https://arxiv.org/abs/2409.00750) · [Seed-TTS 2406.02430](https://arxiv.org/abs/2406.02430) · [CosyVoice 2407.05407](https://arxiv.org/abs/2407.05407) · [CosyVoice 2 2412.10117](https://arxiv.org/abs/2412.10117) · [CosyVoice 3 2505.17589](https://arxiv.org/abs/2505.17589) · [Llasa 2502.04128](https://arxiv.org/abs/2502.04128) · [Spark-TTS 2503.01710](https://arxiv.org/abs/2503.01710) · [IndexTTS2 2506.21619](https://arxiv.org/abs/2506.21619) · [MegaTTS3 2502.18924](https://arxiv.org/abs/2502.18924) · [Vevo 2502.07243](https://arxiv.org/abs/2502.07243) · [VibeVoice 2508.19205](https://arxiv.org/abs/2508.19205) · [MiniMax-Speech 2505.07916](https://arxiv.org/abs/2505.07916) · [Inworld TTS-1 2507.21138](https://arxiv.org/abs/2507.21138)

**HuggingFace model kartalari:** [F5-TTS](https://huggingface.co/SWivid/F5-TTS) · [MaskGCT](https://huggingface.co/amphion/MaskGCT) · [Vevo](https://huggingface.co/amphion/Vevo) · [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) · [IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) · [MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3) · [VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) · [VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) · [higgs-audio-v2](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) · [csm-1b](https://huggingface.co/sesame/csm-1b) · [marvis-tts-250m](https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.1) · [orpheus-3b](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) · [Zonos-v0.1-transformer](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) · [chatterbox](https://huggingface.co/ResembleAI/chatterbox) · [openaudio-s1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini) · [fish-speech-1.5](https://huggingface.co/fishaudio/fish-speech-1.5) · [OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2) · [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) · [Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) · [Llasa-3B](https://huggingface.co/HKUSTAudio/Llasa-3B) · [Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B) · [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · [parler-tts-large-v1](https://huggingface.co/parler-tts/parler-tts-large-v1) · [maya1](https://huggingface.co/maya-research/maya1) · [supertonic](https://huggingface.co/Supertone/supertonic)

**Repolar / LICENSE fayllari:** [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) · [tortoise-tts](https://github.com/neonbjb/tortoise-tts) · [VALL-E-X](https://github.com/Plachtaa/VALL-E-X) · [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) · [F5-TTS](https://github.com/SWivid/F5-TTS) · [FireRedTTS](https://github.com/FireRedTeam/FireRedTTS) · [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) · [index-tts (+ LICENSE)](https://github.com/index-tts/index-tts/blob/main/LICENSE) · [higgs-audio](https://github.com/boson-ai/higgs-audio) · [VibeVoice](https://github.com/microsoft/VibeVoice) · [csm](https://github.com/SesameAILabs/csm) · [chatterbox](https://github.com/resemble-ai/chatterbox) · [Zonos](https://github.com/Zyphra/Zonos) · [fish-speech](https://github.com/fishaudio/fish-speech) · [coqui-ai/TTS](https://github.com/coqui-ai/TTS) · [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval)

**Tijorat hujjatlari:** [ElevenLabs voice cloning](https://elevenlabs.io/docs/product-guides/voices/voice-cloning) · [Cartesia clone voices](https://docs.cartesia.ai/build-with-cartesia/capability-guides/clone-voices) · [Resemble clone overview](https://docs.resemble.ai/voice-creation/voices/clone-overview) · [PlayHT API](https://docs.play.ht/reference/api-getting-started) · [Google Chirp 3 HD](https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd) · [Google Chirp 3 Instant Custom Voice](https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice) · [Hume voice cloning blog](https://www.hume.ai/blog/how-to-clone-your-voice-with-ai) · [OpenAI Voice Engine](https://community.openai.com/t/voice-engine-navigating-the-challenges-and-opportunities-of-synthetic-voices/701714)

**Leaderboardlar:** [Artificial Analysis Speech Arena](https://artificialanalysis.ai/text-to-speech/leaderboard) · [Artificial Analysis Controlled Voice](https://artificialanalysis.ai/text-to-speech/leaderboard/controlled-voice) · TTS Arena V2 (`TTS-AGI/TTS-Arena-V2`) — **olinmadi, HTTP 401**

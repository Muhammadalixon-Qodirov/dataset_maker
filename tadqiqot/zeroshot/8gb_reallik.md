# 8 GB VRAM reallik: RTX 3050 + Windows 11 da zero-shot voice cloning TTS

**Tadqiqot sanasi:** 2026-10-01
**Maqsadli konfiguratsiya:** Windows 11, NVIDIA RTX 3050 **8 GB VRAM**, 16 CPU yadro, **15.9 GB tizim RAM**, torch 2.6.0+cu124
**Bu hujjatning doirasi:** FAQAT apparat reallik va Windows'ga o'rnatilish imkoniyati. Arxitektura, model katalogi, training yo'llari/narxlari va o'zbek tili masalalari boshqa hujjatlarda.

---

## 0. Eng muhim ogohlantirishlar (oldindan o'qing)

### 0.1. Manbalarning aksariyati VRAM raqamini NASHR QILMAYDI

Bu tadqiqotning eng muhim topilmasi. Quyidagi rasmiy repo/model kartalarining HECH BIRI minimal VRAM raqamini e'lon qilmagan:

- XTTS-v2 — "Neither the fork README nor the XTTS docs publish a minimum RAM or VRAM figure" ([localaimaster](https://localaimaster.com/models/coqui-tts))
- F5-TTS — "The project publishes no minimum" ([localaimaster](https://localaimaster.com/blog/f5-tts-setup-guide))
- CosyVoice 2 — rasmiy repoda VRAM raqami yo'q ([GitHub](https://github.com/FunAudioLLM/CosyVoice))
- OpenAudio S1-mini — model kartasida VRAM raqami yo'q ([HF](https://huggingface.co/fishaudio/openaudio-s1-mini/blob/main/README.md))
- MaskGCT — VRAM raqami yo'q ([HF](https://huggingface.co/amphion/MaskGCT), [GitHub](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct))
- IndexTTS-2 — VRAM raqami yo'q ([GitHub](https://github.com/index-tts/index-tts))
- Chatterbox — VRAM raqami yo'q ([GitHub](https://github.com/resemble-ai/chatterbox))
- Spark-TTS — VRAM raqami yo'q ([GitHub](https://github.com/SparkAudio/Spark-TTS))
- Llasa-3B — VRAM raqami yo'q ([HF](https://huggingface.co/HKUSTAudio/Llasa-3B))
- OpenVoice v2 — VRAM raqami yo'q ([GitHub](https://github.com/myshell-ai/OpenVoice))
- Kokoro-82M — VRAM raqami yo'q ([HF](https://huggingface.co/hexgrad/Kokoro-82M))

Bir manba buni to'g'ridan-to'g'ri tasdiqlaydi: **"Not one of the TTS model cards below publishes a VRAM requirement. We checked."** ([localaimaster](https://localaimaster.com/blog/voice-ai-vram-requirements-by-gpu))

Natija: quyidagi jadvaldagi raqamlarning ko'pi **jamoatchilik hisobotlari (community reports)** yoki **uchinchi tomon kalkulyatorlari**. Har biri yorliqlangan.

### 0.2. Windows'dagi "yashirin qurol": CUDA Sysmem Fallback Policy

Windows'da NVIDIA drayveri VRAM tugaganda tizim RAM'ga to'kib ketishi mumkin (Linux'da bunday emas). Bu XTTS fine-tune'ni 8 GB kartada ishlatgan hisobot:

> "I now on 3070 8Gb easilly finetuning. VRAM using show up to 17Gb in Task Manager."
> — [coqui-ai/TTS Discussion #3268](https://github.com/coqui-ai/TTS/discussions/3268) (community report)

AllTalk rasmiy qo'llanmasi ham buni tasdiqlaydi:
> "Windows: Can use system RAM as extended VRAM." / "Linux: Limited to physical VRAM, requiring higher VRAM for reliable operation."
> — [AllTalk XTTS Finetuning Guide](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version))

**LEKIN** — bu yo'l tizim RAM'ni talab qiladi, va aynan shu joyda sizning konfiguratsiyangiz qulaydi (0.3-bo'limga qarang).

### 0.3. 15.9 GB RAM — bu asosiy to'siq, VRAM emas

AllTalk rasmiy talablari:
> "NVIDIA GPU with 12GB+ VRAM (**8GB possible but requires 24GB+ system RAM**)"
> "System RAM: 16GB absolute minimum, 24GB+ recommended"
> — [AllTalk Wiki](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version))

Sizda 15.9 GB bor — ya'ni **"16GB absolute minimum"dan ham pastda** va 8 GB VRAM yo'li uchun talab qilingan 24 GB'dan ancha past. Windows 11 o'zi 3–4 GB RAM iste'mol qiladi.

Boshqa manbalar ham 16 GB'ni minimal, 32 GB'ni tavsiya etilgan deb beradi:
- GPT-SoVITS: "Minimum: 16GB RAM supports basic model inference... Recommended: 32GB RAM for model fine-tuning" ([gpt-sovits.org](https://gpt-sovits.org/gpt-sovits-system-requirements-complete-setup/))
- F5-TTS: RAM "Minimum: 16GB / Recommended: 32GB / Optimal: 64GB" ([clore.ai](https://docs.clore.ai/guides/audio-and-voice/f5-tts))
- Chatterbox: "RAM (Recommended): 16 GB" ([clore.ai](https://docs.clore.ai/guides/audio-and-voice/chatterbox-tts))

### 0.4. RTX 3050 — Ampere, ya'ni FP8/FP4 YO'Q

| Xususiyat | RTX 3060 (Ampere) | RTX 4060 (Ada) | RTX 5060 (Blackwell) |
|---|---|---|---|
| FP8 native | ❌ | ✅ | ✅ |
| FP4 native | ❌ | ❌ | ✅ |
| Tensor Core gen | 3rd | 4th | 5th |

— [TTS-LAB GPU_QA_REFERENCE.md](https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md) (community reference)

RTX 3050 ham Ampere. Ya'ni Orpheus FP8 (1.6 GB) yoki FP4 (0.9 GB) kabi "sig'dirish hiylalari" sizda **ishlamaydi** — faqat bf16/fp16 va (agar Linux'da bo'lsa) int8.

---

## 1. MASTER JADVAL

Ustunlar: model | inference VRAM | fine-tune VRAM | 8 GB da ishlaydimi? | Windows native? | manba

| Model | Inference VRAM | Fine-tune VRAM | 8 GB da? | Windows native? | Manba |
|---|---|---|---|---|---|
| **Piper** | < 100 MB (CPU) | ma'lumot topilmadi | ✅ Ha (CPU'da ham) | ✅ Ha | [codesota](https://www.codesota.com/guides/tts-models) (3rd party) |
| **Kokoro-82M** | ~0.5–1 GB / "< 1 GB" | ma'lumot topilmadi | ✅ Ha | ⚠️ espeak-ng DLL kerak | [localaimaster](https://localaimaster.com/blog/voice-ai-vram-requirements-by-gpu), [codesota](https://www.codesota.com/guides/tts-models) |
| **F5-TTS** (336M) | ~1.5–2.5 GB ⚠️ / ~4 GB ⚠️ (ziddiyat) | 12 GB da ishlagan; 16 GB+ tavsiya; 8 GB uchun **hujjatlashtirilgan config topilmadi** | ✅ Inference ha / ❌ fine-tune yo'q | ✅ Ha | [localaimaster](https://localaimaster.com/blog/voice-ai-vram-requirements-by-gpu), [codesota](https://www.codesota.com/guides/tts-models), [Disc #769](https://github.com/SWivid/F5-TTS/discussions/769) |
| **XTTS-v2** (467M) | ~2 GB (fp16) ⚠️ / ~2.5–3.5 GB / ~4–6 GB (ziddiyat) | Win: 12 GB+ (8 GB + 24 GB RAM bilan mumkin); Linux: 16 GB+ | ✅ Inference ha / ⚠️ fine-tune faqat Sysmem fallback bilan | ✅ Ha (VC++ build tools; prebuilt wheels v0.24.2+) | [AllTalk Wiki](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version)), [Disc #3268](https://github.com/coqui-ai/TTS/discussions/3268) |
| **Chatterbox** (0.5B / Turbo 350M / Nano 110M) | ~2–3 GB (fp16) / 5–7 GB (ziddiyat); min 6 GB, tavsiya 10 GB+ | **LoRA: ≥18 GB** · **GRPO: ≥12 GB** | ✅ Inference ha / ❌ fine-tune yo'q | ⚠️ Rasmiy: "Python 3.11 on Debian 11"; Windows aytilmagan | [chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming), [clore.ai](https://docs.clore.ai/guides/audio-and-voice/chatterbox-tts) |
| **GPT-SoVITS** | **≥6 GB** (rasmiy-ga yaqin) | **6 GB → batch_size=1** ishlaydi; DPO uchun 12 GB+ | ✅ **HA — ikkisi ham** | ✅ **Ha, one-click .7z paket** | [aihub docs](https://docs.aihub.gg/tts/gpt-sovits/), [GitHub](https://github.com/RVC-Boss/GPT-SoVITS) |
| **CosyVoice 2** (0.5B) | ~6–8 GB full / fp16 ~1.2 GB / Q4 ~0.6–0.8 GB (**katta ziddiyat**) | ma'lumot topilmadi | ⚠️ Ehtimol ha | ❌ **pynini conda'dan majburiy** | [localops](https://localops.tech/model/cosyvoice-2-instruct), [madebyagents](https://www.madebyagents.com/models/cosyvoice-2-0), [GitHub](https://github.com/FunAudioLLM/CosyVoice) |
| **Fish-Speech / OpenAudio S1-mini** (0.5B) | **4–6 GB (o'lchangan, RTX 5070)** / 3–4 GB | 8 GB (v1.5/v1.5.1 uchun, 2-darajali manba) | ✅ Ehtimol ha | ⚠️ Community WSL2 ishlatgan | [sneekes.app](https://sneekes.app/posts/fish_speech_installation_guide_for_rtx_5070_on_wsl2/), [nexgpu](https://nexgpu.net/en/models/fish-speech/) |
| **Zonos** | **6 GB+ (rasmiy)**; hybrid uchun 3000-seriya+ | ma'lumot topilmadi | ⚠️ Juda tor | ⚠️ Fork bor (CUDA 12.4 + VS2022 C++) | [Zonos-for-windows](https://github.com/kustomzone/Zonos-for-windows) |
| **IndexTTS-2** | ~6–7 GB / 8 GB tavsiya (3rd party) | ma'lumot topilmadi | ⚠️ Juda tor | ⚠️ pynini + DeepSpeed muammolari (DeepSpeed'ni o'chirish mumkin) | [localaimaster](https://localaimaster.com/blog/voice-ai-vram-requirements-by-gpu), [zimaspace](https://shop.zimaspace.com/pages/index-tts-nvidia-gpu-hardware-requirements), [GitHub](https://github.com/index-tts/index-tts) |
| **VibeVoice-1.5B** | **~6 GB (ComfyUI)** / 5.9 GB fp16 (kalkulyator) / **~7 GB (community o'lchov)** | **LoRA: ≥16 GB** | ⚠️ Chegarada, OOM xavfi yuqori | ⚠️ Windows izohi yo'q | [runcomfy](https://www.runcomfy.com/comfyui-nodes/VibeVoice-ComfyUI), [spheron](https://www.spheron.network/tools/gpu-recommender/microsoft/VibeVoice-1.5B), [VibeVoice-finetuning](https://github.com/voicepowered-ai/VibeVoice-finetuning) |
| **VibeVoice-Large (7B)** | ~20 GB / Q8 ~12 GB | **LoRA: ≥48 GB** | ❌ Yo'q | ⚠️ | [runcomfy](https://www.runcomfy.com/comfyui-nodes/VibeVoice-ComfyUI), [VibeVoice-finetuning](https://github.com/voicepowered-ai/VibeVoice-finetuning) |
| **Orpheus-TTS 3B** | bf16 **6.4 GB** / INT8 3.2 / FP8 1.6 / FP4 0.9 | ma'lumot topilmadi | ⚠️ **"OOM risk" 8 GB kartada** | ❌ vLLM; community "runs on WSL" | [TTS-LAB](https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md), [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) |
| **Higgs Audio v2 / higgs-tts-3-4b** | vaznlar ~10 GB bf16; **min 16 GB, qulay 24 GB**; "8 GB cards aren't enough (even 4-bit struggles in clone mode)" | ma'lumot topilmadi | ❌ **Yo'q** (faqat ComfyUI offload bilan 6 GB + 44 GB RAM) | ❌ | [HF discussion](https://huggingface.co/bosonai/higgs-tts-3-4b/discussions/3), [Issue #29](https://github.com/boson-ai/higgs-audio/issues/29) |
| **MaskGCT** | **ma'lumot topilmadi** | **ma'lumot topilmadi** | ❓ Noma'lum | ⚠️ espeak-ng .msi kerak | [HF](https://huggingface.co/amphion/MaskGCT), [Amphion](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct) |
| **Llasa 1B / 3B / 8B** | **ma'lumot topilmadi** (3B = "4B params", BF16) | Rasmiy: `torchrun --nproc_per_node=8` + DeepSpeed ZeRO-2/3 → ko'p-GPU; Unsloth LoRA 16-bit | ⚠️ 1B ehtimol / 3B-8B yo'q | ❌ DeepSpeed; xcodec2 | [HF](https://huggingface.co/HKUSTAudio/Llasa-3B), [LLaSA_training](https://github.com/zhenye234/LLaSA_training), [Unsloth](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) |
| **Spark-TTS** (0.5B) | **ma'lumot topilmadi** | ❌ **Training code chiqmagan** ("Release the training code" — TODO) | ⚠️ Inference ehtimol ha | ✅ Windows Installation Guide bor | [GitHub](https://github.com/SparkAudio/Spark-TTS) |
| **OpenVoice v2** | **ma'lumot topilmadi** | ma'lumot topilmadi | ❓ (MeloTTS asosida, ehtimol kichik) | ⚠️ unidic/MeCab, monotonic_align kompilyatsiya | [GitHub](https://github.com/myshell-ai/OpenVoice) |

---

## 2. INFERENCE VRAM — batafsil

### 2.1. Diskdagi vazn hajmlari (bu faktlar, VRAM emas)

| Model | Fayl(lar) | Diskda |
|---|---|---|
| Kokoro-82M | bitta .pth | 327 MB |
| F5-TTS | model_1250000.safetensors | 1.35 GB |
| XTTS-v2 | model.pth (1.87 GB) + dvae.pth (0.21 GB) + speaker embeddings (8 MB) | 2.08 GB |
| Chatterbox | T3 backbone + S3Gen vocoder | ~3.2 GB (fp32) |
| VoxCPM-0.5B | pytorch_model.bin + audiovae | 1.6 GB |
| IndexTTS-2 | GPT + S2Mel + Emotion models | 5.9 GB |
| Higgs TTS-3-4B | — | ~10 GB (BF16) |

— [localaimaster](https://localaimaster.com/blog/voice-ai-vram-requirements-by-gpu), [HF Higgs discussion](https://huggingface.co/bosonai/higgs-tts-3-4b/discussions/3)

**Mening o'z baholashim (manba emas):** diskdagi vazn ≠ VRAM. Inference vaqtida vazn + aktivatsiyalar + KV cache + PyTorch allocator fragmentatsiyasi qo'shiladi; amalda vazn hajmining 1.5–2.5 barobari.

### 2.2. Ziddiyatli raqamlar (ikkisini ham beraman)

**XTTS-v2:**
- "At FP16 precision, XTTS-v2 uses approximately **2GB** of VRAM during generation" — [gigagpu](https://gigagpu.com/xtts-v2-vram-requirements/) (3rd party, sahifa 403 qaytardi, qidiruv snippet'idan)
- "Inference VRAM Floor: **~2.5-3.5GB**" — [localaimaster](https://localaimaster.com/blog/voice-ai-vram-requirements-by-gpu)
- "VRAM: **~4 GB**" — [codesota](https://www.codesota.com/guides/tts-models)
- "XTTS-v2 requires **~4–6 GB** VRAM with RTX 4060 (8 GB) as minimum" — [localaimaster](https://localaimaster.com/blog/best-local-tts-models)
- "XTTS (**3.5GB**)" TTS-LAB hisob-kitobida — [TTS-LAB](https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md)
- "~6GB (mentioned in pipeline configuration)" — [spheron](https://www.spheron.network/blog/voice-ai-gpu-infrastructure/)

Diapazon: **2 GB → 6 GB**. Ishonchli xulosa: 8 GB da bemalol sig'adi, lekin "2 GB" optimistik.

**CosyVoice 2 (0.5B) — eng yomon ziddiyat:**
- "approximately **6-8GB VRAM** for standard full-precision inference" — [localops](https://localops.tech/model/cosyvoice-2-instruct)
- "**FP16 requires ~1.2 GB**, Q4_K_M (recommended) requires **~600 MB**" — [madebyagents](https://www.madebyagents.com/models/cosyvoice-2-0)
- "At 4-bit quantization, CosyVoice 2.0 needs about **0.8 GB**" — [madebyagents hardware](https://www.madebyagents.com/models/cosyvoice-2-0/hardware)
- "Q4 quantization requires **4GB** VRAM" — [dev.to](https://dev.to/czmilo/cosyvoice-2025-complete-guide-the-ultimate-multi-lingual-text-to-speech-solution-4l39)
- Rasmiy repo: raqam yo'q — [GitHub](https://github.com/FunAudioLLM/CosyVoice)

Diapazon: **0.6 GB → 8 GB**. 13 barobar farq. Bu raqamlarga ishonmang; o'zingiz o'lchang.

**Fish-Speech / OpenAudio S1-mini:**
- **O'lchangan jamoatchilik hisoboti:** "VRAM Usage: **4-6GB during inference**" (NVIDIA GeForce RTX 5070, WSL2) — [sneekes.app](https://sneekes.app/posts/fish_speech_installation_guide_for_rtx_5070_on_wsl2/)
- "approximately **3-4GB** VRAM for inference" — [spheron](https://www.spheron.network/blog/deploy-open-source-tts-gpu-cloud-2026/)
- "bf16 weights are approximately 1GB, with a **4GB floor** being a reasonable estimate" — [nexgpu](https://nexgpu.net/en/models/fish-speech/)
- Fish Speech **S2-Pro** (katta model): **24 GB** — [nexgpu](https://nexgpu.net/en/models/fish-speech/) → bu sizga mos emas

**VibeVoice-1.5B:**
- "**around 6GB** of VRAM" — [runcomfy](https://www.runcomfy.com/comfyui-nodes/VibeVoice-ComfyUI)
- "FP16: **5.9 GB** / INT8: 2.9 GB / INT4: 1.5 GB" — [spheron](https://www.spheron.network/tools/gpu-recommender/microsoft/VibeVoice-1.5B). **DIQQAT:** bu kalkulyator baholash, manba o'zi aytadi: "Estimated peak VRAM at context length 4,096 and batch size 1, including weights, activations, and KV cache" + "We read the parameter count directly from the model's safetensors metadata... then estimate peak VRAM". Shuningdek bu manba modelni **2.7B params** deb ko'rsatadi (nomi "1.5B" bo'lsa ham).
- "community tests show that generating a multi-speaker dialog with the 1.5B checkpoint consumes approximately **7 GB** of GPU VRAM" — [localaimaster](https://localaimaster.com/blog/vibevoice-local-setup-guide) (community)

8 GB uchun xulosa: 7 GB + Windows desktop ≈ chegara. "Add 10% headroom" tavsiyasi bilan 6.5 GB bo'ladi — **OOM xavfi real**.

### 2.3. 8 GB kartada OOM hujjatlashtirilgan holatlar

**Orpheus 3B bf16 RTX 4060 8 GB'da:**
> "Orpheus bf16 (6.4GB) = **OOM risk ❌**"
> "LLaMA 3.1 8B INT8 (9GB total) = OOM ❌"
> — [TTS-LAB](https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md) (community reference)

**12 GB kartada ham OOM — Higgs Audio v2:**
> "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 MiB. GPU 0 has a total capacity of **11.76 GiB**" (11.55 GiB allocated, 21.19 MiB free)
> — [boson-ai/higgs-audio Issue #29](https://github.com/boson-ai/higgs-audio/issues/29) (community report, 2× RTX 3060 12 GB, Linux)
>
> Tavsiya etilgan workaround: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Maintainer javob bermagan.

**Higgs rasmiy pozitsiya 8 GB haqida:**
> "**8 GB cards aren't enough for the full model (even 4-bit struggles in clone mode)**"
> — [bosonai/higgs-tts-3-4b discussion #3](https://huggingface.co/bosonai/higgs-tts-3-4b/discussions/3)

**RTX 3050 6 GB da XTTS OOM (3rd-party benchmark):**
> "XTTS-v2 shows **OOM on RTX 3050** but 380ms on RTX 4060 8GB"
> — [localaimaster](https://localaimaster.com/blog/best-local-tts-models) (community benchmark, qidiruv snippet'idan)
>
> **Muhim nuans:** bu 3050'ning **6 GB laptop** variantidir, sizdagi 8 GB emas. 8 GB da XTTS ishlashi kerak.

### 2.4. PyTorch allocator xatti-harakati (8 GB da muhim)

> "PyTorch's caching allocator **does not return memory to the driver**, so once a run has peaked, that reservation stays claimed for the process."
> "With 6GB VRAM (like an RTX 3050), you're limited to running **one model at a time with no batching** at fp16."
> — [localaimaster](https://localaimaster.com/blog/best-local-tts-models)

Amaliy oqibat: F5-TTS'da bitta uzun matn generatsiyasi 7 GB peak'ga chiqsa, keyingi barcha generatsiyalar uchun ham o'sha 7 GB band bo'lib turadi. `torch.cuda.empty_cache()` ni har generatsiyadan keyin chaqirish kerak.

### 2.5. VRAM'ni oshiruvchi yashirin xususiyatlar

**F5-TTS:**
> "leaving `--ref_text` empty loads an ASR model ('**extra GPU memory usage**')"
> "generation is **capped at 30 seconds** of audio per pass, so long texts are chunked automatically"
> — [localaimaster F5-TTS guide](https://localaimaster.com/blog/f5-tts-setup-guide)

→ **Har doim `--ref_text` ni qo'lda bering** — Whisper/ASR modelini yuklamasligi uchun. Bu 8 GB da 1–2 GB tejaydi.

**XTTS-v2:**
> "disabling sentence splitting means **more VRAM** and possibly hitting models context length limit"
> — [localaimaster](https://localaimaster.com/models/coqui-tts)

→ Sentence splitting'ni O'CHIRMANG.

---

## 3. FINE-TUNE VRAM — batafsil

### 3.1. XTTS-v2 — 8 GB da hujjatlashtirilgan YAGONA jiddiy yo'l (Windows-ga xos)

**Rasmiy (AllTalk) talablar** — [AllTalk Wiki](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version)):
> Windows: "NVIDIA GPU with **12GB+ VRAM** (8GB possible but requires **24GB+ system RAM**)"
> Linux: "NVIDIA GPU with **16GB+ VRAM** recommended"
> System RAM: "**16GB absolute minimum, 24GB+ recommended**"

**Jamoatchilik hisobotlari** — [coqui-ai/TTS Discussion #3268](https://github.com/coqui-ai/TTS/discussions/3268):

| GPU | Config | Natija |
|---|---|---|
| RTX 3060 12 GB | BATCH_SIZE=3, GRAD_ACCUM_STEPS=84 | ❌ OOM |
| RTX 3060 12 GB | BATCH_SIZE=2, GRAD_ACCUM_STEPS=126 | ❌ OOM |
| RTX 3060 12 GB | BATCH_SIZE=1, GRAD_ACCUM_STEPS=252 | ✅ Ishladi |
| RTX 3070 **8 GB** | Sysmem Fallback yoqilgan | ✅ "easilly finetuning. VRAM using show up to **17Gb** in Task Manager" |
| 8 GB GPU | BATCH_SIZE=1 (drayver yangilashdan oldin) | ❌ OOM → drayver yangilangandan keyin "completed 1 epoch" |

> "The most significant breakthrough mentioned was updating NVIDIA drivers to enable **'CUDA – Sysmem Fallback Policy,'** which allows GPUs to utilize system RAM... appears in recent November drivers under 'Manage 3D settings'"

**Sizning holatingizda:** 17 GB "VRAM" = 8 GB haqiqiy VRAM + ~9 GB tizim RAM'dan to'kilgan. Sizda jami 15.9 GB RAM bor, Windows ~3–4 GB oladi → ~12 GB bo'sh. Nazariy jihatdan 9 GB sig'ishi mumkin, lekin **nol zahira**. Mening baholashim (manba emas): bu ishlashi mumkin, lekin juda sekin (PCIe orqali swap) va beqaror. AllTalk 24 GB RAM talab qilgani bejiz emas.

**Gradient accumulation misoli (rasmiy):**
> "batch_size = 8, gradient_accumulation_steps = 4 # Effective batch size = 32"
> Xotira cheklovi uchun: "Reduce batch size to 2"

### 3.2. GPT-SoVITS — 8 GB da ENG REAL yo'l

Bu yagona model bo'lib, rasmiy hujjat past-VRAM konfiguratsiyasini aniq beradi:

> "You'll require great specs & a NVIDIA GPU with **>=6G VRAM** to run it smoothly."
> **SoVITS training:** Batch size `2` tavsiya; "Use **`1` if the GPU has 6GB VRAM**"
> **GPT training:** Batch size 2 tavsiya; "**1 if your gpu has 6G vram**"
> **DPO training:** "A GPU with **12G VRAM** or more" + batch size 1
> — [AI Hub GPT-SoVITS docs](https://docs.aihub.gg/tts/gpt-sovits/)

VRAM darajalari (3rd party):
> Basic Inference: **6GB** · Moderate Training: **8GB–12GB** · Professional: **16GB+**
> "If your GPU is low on memory, you can turn on **half precision mode (fp16)** in the WebUI"
> — [gpt-sovits.org](https://gpt-sovits.org/gpt-sovits-system-requirements-complete-setup/)

**Xulosa:** 8 GB da GPT-SoVITS'ning ikki bosqichli fine-tune'i (SoVITS + GPT) batch_size=1, fp16 bilan ishlaydi. DPO'ni o'tkazib yuborish kerak (12 GB talab qiladi).

### 3.3. F5-TTS — 8 GB uchun hujjatlashtirilgan config YO'Q

Rasmiy training README'da VRAM yo'riqnomasi yo'q ([manba](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/train/README.md)). Faqat `++datasets.batch_size_per_gpu=19200` misoli.

**Jamoatchilik konfiguratsiyalari** — [Discussion #57](https://github.com/SWivid/F5-TTS/discussions/57), [#769](https://github.com/SWivid/F5-TTS/discussions/769):

| GPU | batch_size_per_gpu | grad_accum | max_samples | 8-bit optim | Natija |
|---|---|---|---|---|---|
| RTX 4090 24 GB | 4000 | 78 | — | — | ✅ "working very well and also with very big dataset" |
| RTX 3090 24 GB | 1618 (frame) | 1 | 64 | — | ✅ 60–80 soatlik dataset, 11 epoch |
| RTX 4090 (ispan) | 4000 | — | — | — | ✅ "6h 50m 53s for 4,000 batch size and 93000 batches" |
| RTX 4080 Laptop **12 GB** | "I assume **6400** should work with 12GB VRAM" | — | ≤8 ("any value for max samples higher than 8 leads to horribly slow training") | ❌ yoqilmagan | ✅ 320 epoch |
| RTX 4070 Laptop | 2200 | 1 | 64 | ❌ "8 bit optimizer: **false** (not enabled)" | ✅ 150 epoch / 30k step |

> "No configurations for 8GB, 10GB, or 12GB VRAM were explicitly documented" (#57 uchun)

3rd-party talablar:
> "F5-TTS requires any NVIDIA GPU with **16GB+ VRAM**, with **24GB** (RTX 3090, 4090) being comfortable, and is one of the few fine-tunable TTS models that can fit on **16GB with reduced batch size**" — qidiruv natijasi sintezi (3rd party)
> clore.ai: Minimum "RTX 3060 12GB" / **6GB** VRAM; Recommended "RTX 4080 16GB" / 12GB; Optimal "RTX 4090 24GB" / 16GB — [clore.ai](https://docs.clore.ai/guides/audio-and-voice/f5-tts)

**LoRA yo'li mavjud:**
> "Full fine-tune is currently supported, while **LoRA or adapter methods are not yet implemented in the official F5-TTS repository**. However, LoRA fine-tuning reduces VRAM usage and produces tiny checkpoint files (**~6 MB vs ~2.7 GB** for full fine-tuning)."
> — [instavar/f5-tts-lora-finetuning](https://github.com/instavar/f5-tts-lora-finetuning) (jamoatchilik fork, "GPU-validated on IMDA NSC")
>
> LoRA uchun aniq VRAM raqami: **ma'lumot topilmadi**.

**RAM ogohlantirish:** F5-TTS training `num_workers` default = 16.
> Bir foydalanuvchi 500 GB RAM'li superkompyuterda ham: "it still run out of memory every time". Tavsiya: "Reducing `num_workers` parameter (default 16) to **1** was recommended to lower RAM consumption during training."
> — [Discussion #57](https://github.com/SWivid/F5-TTS/discussions/57)
>
> Sizdagi 15.9 GB RAM uchun bu **majburiy sozlama**: `num_workers=1` yoki 2.

### 3.4. Chatterbox — 8 GB da imkonsiz

> **LoRA:** "You will need a CUDA gpu with **at least 18gb of vram** depending on your dataset size and training params."
> **GRPO:** "You will need a CUDA gpu with **at least 12gb of vram** depending on your dataset size and training params."
> — [davidbrowne17/chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming) (jamoatchilik fork — rasmiy repo fine-tune'ni qo'llab-quvvatlamaydi)

Rasmiy repo: "The documentation does **not officially support finetuning** for end users." — [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)

### 3.5. VibeVoice — 8 GB da imkonsiz

> "To train a VibeVoice **1.5B** LoRa, a machine with **at least 16gb VRAM** is recommended."
> "To train a VibeVoice **7B** LoRa, a machine with **at least 48gb VRAM** is recommended."
> "Keep in mind **longer audios increase VRAM requirements**"
> — [voicepowered-ai/VibeVoice-finetuning](https://github.com/voicepowered-ai/VibeVoice-finetuning)

Misol konfiguratsiya: `--per_device_train_batch_size 8 --gradient_accumulation_steps 16 --bf16 True --gradient_checkpointing False`.
Gradient checkpointing mavjud: "A memory-saving technique that trades compute for memory." bf16: "speeding up training and reducing memory on compatible GPUs (**Ampere series and newer**)" → RTX 3050 Ampere, bf16 ishlaydi.

### 3.6. Fish-Speech / OpenAudio

Rasmiy finetune hujjatida VRAM raqami yo'q:
> "You can adjust `--num-workers` and `--batch-size` to increase extraction speed, but please make sure not to exceed your GPU memory limit."
> — [speech.fish.audio/finetune](https://speech.fish.audio/finetune/)

2-darajali manba: "Official documentation specifies **8GB for fine-tuning** on Fish Speech v1.5/v1.5.1" — [spheron deployment guide](https://www.spheron.network/blog/deploy-open-source-tts-gpu-cloud-2026/). **Bu raqamni rasmiy hujjatda tasdiqlay olmadim.**

### 3.7. Llasa — ko'p-GPU training

> Training buyrug'i: `torchrun --nproc_per_node=8 train_tts.py config.json`
> DeepSpeed configlar: `ds_config_zero2.json`, `ds_config_zero3.json`, SLURM script
> — [zhenye234/LLaSA_training](https://github.com/zhenye234/LLaSA_training)

8 GPU + DeepSpeed ZeRO. Consumer GPU uchun raqam: **ma'lumot topilmadi**.

Unsloth TTS yo'li (LoRA):
> "Sesame-CSM (1B), Orpheus-TTS (3B), Spark-TTS (0.5B), **Llasa-TTS (1B)**, Oute-TTS (1B)" notebooklari mavjud
> "In all our TTS notebooks, we enable **LoRA (16-bit)** training and disable QLoRA (4-bit) training with: `load_in_4bit = False`"
> "on a Colab T4 GPU, a few epochs on 3h of data may take 1-2 hours"
> — [Unsloth TTS docs](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning)
>
> Unsloth hujjatida **per-model VRAM raqami YO'Q** — "No minimum VRAM figures (like 8GB) or per-model VRAM requirements appear in this documentation."
> Colab T4 = 16 GB → bu de-fakto baseline.

3rd party: "For 3B-4B models using LoRA with 16-bit precision, approximately **10 GB** of VRAM is required" — [unsloth/dev.to sintezi](https://dev.to/jangwook_kim_e31e7291ad98/fine-tune-llms-with-lora-and-qlora-2026-guide-33lf). → **Llasa-1B LoRA 16-bit ehtimol 8 GB'ga sig'adi** (mening baholashim, manba emas).

### 3.8. Fine-tune imkonsiz / ma'lumot yo'q bo'lganlar

| Model | Holat |
|---|---|
| Spark-TTS | ❌ "Release the training code" — TODO ro'yxatida; training kodi chiqmagan — [GitHub](https://github.com/SparkAudio/Spark-TTS) |
| IndexTTS-2 | Fine-tune VRAM: **ma'lumot topilmadi** |
| MaskGCT | **ma'lumot topilmadi** |
| CosyVoice 2 | **ma'lumot topilmadi** (rasmiy repoda training yo'riqnomasi bor, lekin VRAM raqami yo'q) |
| Higgs Audio v2 | **ma'lumot topilmadi** |
| Zonos | **ma'lumot topilmadi** |
| OpenVoice v2 | **ma'lumot topilmadi** |
| Kokoro / Piper | Zero-shot cloning modeli emas; fine-tune VRAM **ma'lumot topilmadi** |

### 3.9. Xotira tejash hiylalari — nimalar hujjatlashtirilgan

| Hiyla | Holat Windows'da | Manba |
|---|---|---|
| **Gradient accumulation** | ✅ Ishlaydi, hamma joyda | XTTS: GRAD_ACCUM_STEPS=252; F5-TTS: grad_accumulation_steps=78 |
| **Batch size = 1** | ✅ Ishlaydi | GPT-SoVITS 6 GB; XTTS 3060 12 GB |
| **fp16 / bf16 mixed precision** | ✅ Ishlaydi (3050 = Ampere, bf16 ham bor) | VibeVoice: "--bf16 True... Ampere series and newer"; GPT-SoVITS: WebUI'da fp16 toggle |
| **Gradient checkpointing** | ✅ Ishlaydi | VibeVoice: `--gradient_checkpointing` flagi |
| **CUDA Sysmem Fallback Policy** | ✅ **FAQAT Windows'da** — RAM'ni VRAM sifatida ishlatadi | [Disc #3268](https://github.com/coqui-ai/TTS/discussions/3268) |
| **Reduced segment / max_samples** | ✅ | F5-TTS: "max samples higher than 8 leads to horribly slow training" |
| **num_workers kamaytirish** | ✅ RAM tejaydi (VRAM emas) | F5-TTS: 16 → 1 |
| **8-bit optimizer (bitsandbytes)** | ❌ **Windows native'da ISHLAMAYDI** (4-bo'limga qarang) | [bitsandbytes Windows caveats](https://theneuralbase.com/bitsandbytes/learn/beginner/windows-support-caveats/) |
| **DeepSpeed ZeRO offload** | ❌ **Windows'da o'rnatilmaydi** (4-bo'limga qarang) | [IndexTTS Issue #393](https://github.com/index-tts/index-tts/issues/393) |
| **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** | ✅ Fragmentatsiyaga qarshi | [higgs-audio Issue #29](https://github.com/boson-ai/higgs-audio/issues/29) |

---

## 4. WINDOWS'GA O'RNATILISH — buzilgan paketlar va yechimlar

### 4.1. ❌ bitsandbytes — 8-bit optimizer Windows'da ISHLAMAYDI

Bu sizning holatingiz uchun eng muhim blocker, chunki 8-bit Adam 8 GB kartada training VRAM'ni sezilarli kamaytiradi.

> "bitsandbytes provides optimized CUDA kernels for quantization (8-bit and 4-bit), but **these kernels are not compiled for Windows in the default distribution**."
>
> Xato: "CUDA Setup failed with RuntimeError: Unexpected Exception in bnb.cuda_setup(). Compiling the CUDA kernels for 4-bit quantization requires access to CUDA kernels that are **not available on Windows**."
>
> "**Windows support for bitsandbytes exists only for inference with pre-quantized models... for training or quantizing, it's not viable.**"
>
> Rasmiy tavsiya: "**Use WSL2, a Linux VM, or a cloud GPU instead.**"
> — [theneuralbase bitsandbytes Windows caveats](https://theneuralbase.com/bitsandbytes/learn/beginner/windows-support-caveats/)

**Jamoatchilik workaround:** `bitsandbytes-windows` / `bitsandbytes-windows-webui` nomli nosxalangan wheel'lar mavjud — [Mindfire Technology guide](https://www.mindfiretechnology.com/blog/archive/installing-bitsandbtyes-for-windows-so-that-you-can-do-peft/). Bu versiyalar odatda orqada qoladi va yangi torch bilan mos kelmaydi (mening baholashim).

**Tasdiqlovchi fakt:** F5-TTS'ni muvaffaqiyatli fine-tune qilgan **ikki** jamoatchilik hisoboti ham 8-bit optimizer'ni **o'chirgan**: "8 bit optimizer: false (not enabled)" / "Did not enable it" — [Disc #769](https://github.com/SWivid/F5-TTS/discussions/769).

### 4.2. ❌ DeepSpeed — Windows'da kompilyatsiya qilinmaydi

> "DeepSpeed (v0.17.5) does **not seem to have any pre-compiled wheels available on PyPI for Windows** with Python 3.12. It always tries to compile from source, which requires installing the full Microsoft C++ Build Tools and CUDA Toolkit to resolve linker errors."

Aniq xatolar — [index-tts Issue #393](https://github.com/index-tts/index-tts/issues/393), [#395](https://github.com/index-tts/index-tts/issues/395), [#392](https://github.com/index-tts/index-tts/issues/392):
> `LINK : fatal error LNK1181: cannot open input file 'aio.lib'`
> `ModuleNotFoundError: No module named 'dskernels'`
> `PyTorch version mismatch! DeepSpeed ops were compiled and installed with a different version than what is being used at runtime`
>
> Muhit: Python 3.12 (cp312), PyTorch 2.8.0+cu129 / 2.8.0+cu128 / 2.5.1+cu121, DeepSpeed 0.17.5, Windows 11.
> "**No workaround is documented** in the issue thread." Maintainerlar javob bermagan.

**Yaxshi yangilik — IndexTTS rasmiy yo'riqnomasi DeepSpeed'ni ixtiyoriy deb belgilaydi:**
> "**DeepSpeed** _may_ speed up inference on some systems, but it could also make it slower — it depends on your hardware, drivers and OS."
> "**Windows:** DeepSpeed may be difficult to install. You can **skip it by removing the `--all-extras` flag** and adding the other feature flags manually."
> — [index-tts/index-tts](https://github.com/index-tts/index-tts)

**Kelajak:** DeepSpeed'ga native Windows support PR qilingan ("Add Native Windows Support... CPU-side extension ops building and running under MSVC, and fixes Windows-specific runtime bugs including paths, timers, and MSVC OpenMP issues") — [DeepSpeed PR #8542](https://github.com/deepspeedai/DeepSpeed/pull/8542). Hali merge/release holatini tekshirish kerak.

### 4.3. ❌ pynini / WeTextProcessing — pip bilan Windows'da qurilmaydi

CosyVoice, IndexTTS, ChatTTS, ComfyUI-FunAudioLLM hammasiga ta'sir qiladi.

> "Pynini and webrtcvad **cannot be built on Windows**, and some dependencies cannot be installed via pip on Windows, such as pynini."
> Xato: `ERROR: Failed building wheel for pynini`
> — [ComfyUI-FunAudioLLM Issue #7](https://github.com/SpenserCai/ComfyUI-FunAudioLLM/issues/7), [ChatTTS Issue #162](https://github.com/2noise/ChatTTS/issues/162)

**Hujjatlashtirilgan yechim (ishlaydi)** — [index-tts Issue #61](https://github.com/index-tts/index-tts/issues/61):
```
conda create -n index-tts python=3.10
conda activate index-tts
conda install -y -c conda-forge pynini==2.1.5
pip install WeTextProcessing==1.0.3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```
> "On Windows, the pynini module **can only be installed using conda**, which is why conda is recommended for Windows."

Shuningdek `AssertionError: Torch not compiled with CUDA enabled` xatosi — to'g'ri CUDA wheel index'ini ko'rsatish bilan hal bo'ladi.

**Muhim:** siz torch 2.6.0+cu124 ishlatasiz. Conda muhitida pynini o'rnatgandan keyin torch'ni o'z versiyangizga qaytib o'rnatish kerak, aks holda conda boshqa torch tortib keladi.

### 4.4. ⚠️ espeak-ng — DLL topilmaydi

MaskGCT, Kokoro, Zonos, StyleTTS2, ba'zi F5-TTS yo'llariga ta'sir qiladi.

> MaskGCT rasmiy: "**For Windows:** Please visit https://github.com/espeak-ng/espeak-ng/releases to download **.msi installer**"
> — [Amphion MaskGCT](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)

Xato: `Exception: [!] No espeak backend found. Install espeak-ng or espeak to your system.` — [coqui-ai/TTS Discussion #1973](https://github.com/coqui-ai/TTS/discussions/1973)

**Yechimlar** — [Phonemizer install docs](https://bootphon.github.io/phonemizer/install.html):
1. Python kodida: `EspeakWrapper.set_library('C:\Program Files\eSpeak NG\libespeak-ng.dll')`
2. Muhit o'zgaruvchisi: `PHONEMIZER_ESPEAK_LIBRARY="C:\Program Files\eSpeak NG\libespeak-ng.dll"`
3. PowerShell: `$Env:PHONEMIZER_ESPEAK_PATH='C:\Program Files\eSpeak NG\espeak-ng.exe'` + `$Env:PHONEMIZER_ESPEAK_LIBRARY='C:\Program Files\eSpeak NG\libespeak-ng.dll'`
4. Conda: `conda env config vars set PHONEMIZER_ESPEAK_LIBRARY="..."`

> "Make sure you install the correct .msi file for your platform (**X64** or X86)... if unsure, try `espeak-ng-X64.msi` first."

Zonos uchun qiziq nuans: upstream repo espeak-ng'ni faqat Linux bo'limida ko'rsatadi ("apt install -y espeak-ng"), Windows bo'limida espeak haqida hech narsa yo'q — [Zonos-for-windows](https://github.com/kustomzone/Zonos-for-windows).

### 4.5. ⚠️ flash-attn

> "it **might work for Windows starting v2.3.2** but Windows compilation still requires more testing"
> "FlashAttention-2 with CUDA supports **Ampere, Ada, or Hopper** GPUs"
> — [Dao-AILab/flash-attention](https://github.com/dao-ailab/flash-attention)

RTX 3050 = Ampere → arxitektura jihatdan mos, lekin Windows'da kompilyatsiya qilish og'riqli. Aksariyat TTS repolari `sdpa` yoki `eager` attention'ga fallback qiladi (VibeVoice ComfyUI: "Choose from various attention types" — [runcomfy](https://www.runcomfy.com/comfyui-nodes/VibeVoice-ComfyUI)).

Windows uchun prebuilt triton/sage-attention/flash-attention wheel yo'riqnomasi mavjud: [civitai guide](https://civitai.com/articles/9555/installing-triton-and-sage-attention-flash-attention-and-x-formers-win) (jamoatchilik).

### 4.6. ⚠️ HuggingFace cache symlink muammosi (Windows)

Rasmiy HF hujjati — [manage-cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache):
> "In order to have an efficient cache-system, `huggingface-hub` uses symlinks. However, **symlinks are not supported on all machines. This is a known limitation especially on Windows.** When this is the case, `huggingface_hub` does not use the `blobs/` directory but directly stores the files in the `snapshots/` directory instead... However, the cache-system is **less efficient as a single file might be downloaded several times** if multiple revisions of the same repo are downloaded."
>
> "If you want to benefit from the symlink-based cache-system on a Windows machine, you either need to **activate Developer Mode** or to run **Python as an administrator**."
>
> Ogohlantirish xabarini o'chirish: `HF_HUB_DISABLE_SYMLINKS_WARNING=true`
> Symlink'siz rejimni majburan yoqish: `HF_HUB_DISABLE_SYMLINKS=1`

**Amaliy ta'sir:** IndexTTS-2 (5.9 GB), Higgs (10 GB) kabi modellar har revision uchun qayta yuklab olinadi → disk tez to'ladi. Sizning MEMORY.md'da C: diski to'lgani va `HF_HOME=D:\hf_cache` ishlatilgani qayd etilgan — **xuddi shu sozlamani davom ettirish kerak**.

### 4.7. ✅ Windows one-click / portable buildlar (ENG YAXSHI YO'L)

| Loyiha | Windows paketi | Tafsilot | Manba |
|---|---|---|---|
| **GPT-SoVITS** | ✅ **Integrated .7z + `go-webui.bat`** | "Windows users (tested with Windows 10 and above) can download an integrated package and **double-click on go-webui.bat**". Paket: Python + PyTorch CUDA runtimes + UVR5 vocal splitter + ASR modellari, **~12.8 GB** (7z). Misol: `GPT-SoVITS-v3lora-20250228.7z` | [GitHub](https://github.com/RVC-Boss/GPT-SoVITS), [pasqualepillitteri](https://pasqualepillitteri.it/en/news/7051/gpt-sovits-voice-cloning-guide) |
| **xtts-finetune-webui** | ✅ `install.bat` / `start.bat` | "First start `install.bat`... To start the server start `start.bat`... Go to `127.0.0.1:5003`" | [daswer123](https://github.com/daswer123/xtts-finetune-webui) |
| **Zonos-for-windows** | ✅ Auto-installer | "Auto installs and downloads models (6G vram min)". Talablar: CUDA 12.4, VS 2022 C++ compiler, `Set-ExecutionPolicy Unrestricted` | [kustomzone](https://github.com/kustomzone/Zonos-for-windows) |
| **zonos-windows-tts** | ✅ Windows-optimized setup | "automated installation, phonemizer integration, and GPU memory management" | [Foadsf](https://github.com/Foadsf/zonos-windows-tts) |
| **AllTalk TTS** | ✅ Windows finetune UI | XTTS finetuning uchun to'liq Windows yo'riqnomasi | [erew123](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version)) |
| **Spark-TTS** | ✅ "Windows Installation Guide" mavjud | "If you're on Windows, please refer to the Windows Installation Guide" | [GitHub](https://github.com/SparkAudio/Spark-TTS) |
| **coqui-tts fork (XTTS)** | ✅ Prebuilt wheels | "Visual C++ build tools are required; the maintained fork has **published prebuilt wheels since v0.24.2**" | [localaimaster](https://localaimaster.com/models/coqui-tts) |

GPT-SoVITS qo'shimcha Windows talablari:
> "Download and place **ffmpeg.exe and ffprobe.exe in the GPT-SoVITS root**" va "Install **Visual Studio 2017**"
> — [GitHub README](https://github.com/RVC-Boss/GPT-SoVITS)

Tasdiqlangan Python/PyTorch kombinatsiyalari (rasmiy): Python 3.10 + PyTorch 2.5.1 + CUDA 12.4; Python 3.11 + PyTorch 2.5.1 + CUDA 12.4; Python 3.11 + PyTorch 2.7.0 + CUDA 12.8; Python 3.9 + PyTorch 2.2.2 (CPU).
→ Sizdagi **torch 2.6.0+cu124** rasmiy tasdiqlangan ro'yxatda yo'q, lekin 2.5.1+cu124 va 2.7.0+cu128 orasida — ehtimol ishlaydi (mening baholashim).

### 4.8. ❌ WSL2 / Linux talab qiladiganlar

| Model | Dalil |
|---|---|
| **Fish-Speech / OpenAudio** | Jamoatchilik qo'llanmasi RTX 5070 uchun **WSL2** ishlatgan — [sneekes.app](https://sneekes.app/posts/fish_speech_installation_guide_for_rtx_5070_on_wsl2/) (nega native Windows ishlamaganini tushuntirmaydi) |
| **Orpheus-TTS** | vLLM asosida ("uses vllm under the hood"); jamoatchilik: "Gradio WebUI that runs smoothly on **WSL** and CUDA" — [GitHub](https://github.com/canopyai/Orpheus-TTS) |
| **Chatterbox** | Rasmiy: "We developed and tested Chatterbox on **Python 3.11 on Debian 11 OS**" — Windows haqida hech narsa yo'q. Streaming fork faqat `.venv/bin/activate` (Linux yo'li) ko'rsatadi — [resemble-ai](https://github.com/resemble-ai/chatterbox), [chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming) |
| **CosyVoice** | Rasmiy repo faqat Ubuntu/CentOS: "ubuntu: sudo apt-get install sox libsox-dev" / "centos: sudo yum install sox sox-devel". Windows yo'riqnomasi yo'q — [GitHub](https://github.com/FunAudioLLM/CosyVoice) |
| **Llasa training** | `torchrun --nproc_per_node=8` + DeepSpeed → Linux ko'p-GPU — [LLaSA_training](https://github.com/zhenye234/LLaSA_training) |
| **Har qanday bitsandbytes training** | "Use WSL2, a Linux VM, or a cloud GPU instead" — [bitsandbytes caveats](https://theneuralbase.com/bitsandbytes/learn/beginner/windows-support-caveats/) |

### 4.9. monotonic_align va long-path

- **monotonic_align:** VITS-dan kelib chiqqan repolarda (GPT-SoVITS, MeloTTS/OpenVoice, Bert-VITS2) Cython kengaytmasi. Windows'da aniq buzilish/yechim hujjati **ma'lumot topilmadi**. Lekin GPT-SoVITS rasmiy ravishda "Install Visual Studio 2017" talab qiladi — **mening baholashim (manba emas):** bu aynan shu kompilyatsiya uchun. Windows one-click paketi buni hal qilib bergan.
- **Long path limit (260 belgi):** TTS loyihalari uchun aniq hujjatlashtirilgan blocker **ma'lumot topilmadi**. Umumiy tavsiya (mening baholashim): `HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1` va loyihalarni `C:\tts\` kabi qisqa yo'lda saqlash. Sizning hozirgi yo'lingiz (`OneDrive\Ishchi stol\New folder (4)\...`) juda uzun va **OneDrive sinxronizatsiyasi** ham model fayllarini sekinlashtiradi/buzadi — modellarni OneDrive'dan TASHQARIDA saqlash kerak.

---

## 5. TEZLIK — o'lchangan raqamlar

### 5.1. RTF (real-time factor) — GPU

| Model | GPU | RTF | Manba |
|---|---|---|---|
| **GPT-SoVITS v2 ProPlus** | RTX 4060 Ti | **0.028** | [rasmiy README](https://github.com/RVC-Boss/GPT-SoVITS) |
| **GPT-SoVITS v2 ProPlus** | RTX 4090 | **0.014** | [rasmiy README](https://github.com/RVC-Boss/GPT-SoVITS) |
| **IndexTTS-2 fp16** | RTX 4090 | **0.3257** | [rasmiy README](https://github.com/index-tts/index-tts) |
| **IndexTTS-2 fp32** | RTX 4090 | **0.3748** | [rasmiy README](https://github.com/index-tts/index-tts) |
| **Kokoro-82M** | RTX 4090 | **0.04** | [spheron](https://www.spheron.network/blog/voice-ai-gpu-infrastructure/) |
| **Kokoro-82M** | RTX 5090 | **0.025** | [spheron](https://www.spheron.network/blog/voice-ai-gpu-infrastructure/) |
| **XTTS v2** | RTX 4090 | **0.30** | [spheron](https://www.spheron.network/blog/voice-ai-gpu-infrastructure/) |
| **Zonos** | RTX 4090 | "real-time factor of **~2x**" (1 sek hisoblashda 2 sek audio) | [Zonos](https://github.com/kustomzone/Zonos-for-windows) |
| **Orpheus 3B bf16** | RTX 3060 12 GB | **~0.85** | [TTS-LAB](https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md) (community) |
| **Orpheus 3B INT8/FP8** | RTX 4060 Ti 16 GB | ~0.55 | [TTS-LAB](https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md) |
| **Orpheus 3B FP4** | RTX 5060 8 GB | ~0.48 | [TTS-LAB](https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md) |
| **Dia-1.6B** | RTX 4060 Ti 16 GB / 5060 8 GB | ~0.55 | [TTS-LAB](https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md) |
| **Chatterbox** | RTX 4060 Ti 16 GB | **0.12** | [localaimaster](https://localaimaster.com/blog/best-local-tts-models) (community benchmark) |
| **F5-TTS** | RTX 4060 Ti 16 GB | **0.14** | [localaimaster](https://localaimaster.com/blog/best-local-tts-models) |
| **XTTS-v2** | RTX 4060 Ti 16 GB | **0.30** | [localaimaster](https://localaimaster.com/blog/best-local-tts-models) |
| **Chatterbox (streaming fork)** | ma'lumot topilmadi (GPU aytilmagan) | **0.499**, "Latency to first chunk: 0.472s", "Total generation time: 2.915s / Total audio duration: 5.840s" | [chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming) |
| **F5-TTS Base (Vocos), Triton+TRT-LLM** | L20 (datacenter) | **0.0394** (concurrency 2, "Avg Latency 253 ms"); offline TRT-LLM 0.0402; **offline PyTorch 0.1467** | [rasmiy README](https://github.com/SWivid/F5-TTS) (26 prompt pairs, 16 NFE) |
| **Spark-TTS, Triton** | L20 | **0.1362**, "Avg Latency 876.24 ms" (concurrency 1) | [rasmiy README](https://github.com/SparkAudio/Spark-TTS) |
| **F5-TTS (demo, early)** | ma'lumot topilmadi | "close to **RTF = 3.0**" | [Issue #81](https://github.com/SWivid/F5-TTS/issues/81) (2024-10, community) |

**ZIDDIYAT:** F5-TTS uchun rasmiy PyTorch offline RTF = 0.1467 (L20), lekin 2024-oktyabrdagi issue'da RTF ≈ 3.0. Ikkinchisi juda erta versiya/demo Space edi.

### 5.2. Birinchi audio chunk latensiyasi (community benchmark)

| Model | RTX 3050 **6 GB** | RTX 4060 **8 GB** |
|---|---|---|
| Chatterbox (0.5B) | **280 ms** | 190 ms |
| F5-TTS (336M) | **320 ms** | 220 ms |
| XTTS-v2 | **OOM** | 380 ms |

— [localaimaster](https://localaimaster.com/blog/best-local-tts-models) (community benchmark)

**DIQQAT:** bu 3050'ning **6 GB** variantidir. Sizda 8 GB bor, shuning uchun XTTS OOM bo'lmasligi kerak, lekin tezlik 3050'ning o'zi uchun ko'rsatilgan (Chatterbox 280 ms, F5-TTS 320 ms) — ya'ni **sizning kartangizda ham shu atrofda** bo'lishi kutiladi.

### 5.3. Streaming imkoniyati va birinchi-token latensiyasi

| Model | Streaming? | Latensiya | Manba |
|---|---|---|---|
| **CosyVoice 2** | ✅ bi-directional | "latency as low as **150ms**" | [rasmiy](https://github.com/FunAudioLLM/CosyVoice) |
| **Orpheus-TTS** | ✅ | "**~200ms** streaming latency for realtime applications, reducible to **~100ms** with input streaming" | [rasmiy](https://github.com/canopyai/Orpheus-TTS) |
| **XTTS-v2** | ✅ | "XTTS can stream with **<200ms** latency"; "streaming is typically slower than regular inference, but it allows to get a first chunk of audio faster" | [localaimaster](https://localaimaster.com/models/coqui-tts) |
| **Chatterbox** | ✅ (fork) | "Latency to first chunk: **0.472s**" | [chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming) |
| **F5-TTS** | ✅ chunk-based | 30 sek cap bilan avtomatik chunking | [codesota](https://www.codesota.com/guides/tts-models), [localaimaster](https://localaimaster.com/blog/f5-tts-setup-guide) |
| **Fish Speech** | ✅ | — | [codesota](https://www.codesota.com/guides/tts-models) |
| **Kokoro** | ✅ chunked | RTF 0.03 | [codesota](https://www.codesota.com/guides/tts-models) |
| **Piper** | ✅ sentence-level | RTF 0.008 | [codesota](https://www.codesota.com/guides/tts-models) |
| **IndexTTS-2** | Streaming tadqiqot maqolasi bor ("Faster IndexTTS-2: Accelerating and **Streaming** Autoregressive Zero-Shot TTS on GPUs", [arXiv](https://arxiv.org/pdf/2607.21042)) | — | — |
| MaskGCT / VibeVoice / Higgs / Llasa / Spark-TTS | ma'lumot topilmadi | — | — |

### 5.4. CPU-only tezlik

TTS-LAB benchmark'i (**Intel Xeon D-1528, 6 yadro, 1.90 GHz, Broadwell-DE, 35W TDP**, 19 GB DDR4) — [manba](https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md):

| Engine | CPU RTF | Holat |
|---|---|---|
| **Piper TTS** | **0.37** | ✅ Real-time |
| MeloTTS | 1.01 | ⚠️ Chegarada |
| StyleTTS 2 | 1.52 | ⚠️ Chegarada |
| **Kokoro-82M** | **2.83** | ❌ Sekin |
| **XTTS-v2** | **3.85** | ❌ Sekin |
| Dia-1.6B | 38.88 | ❌ Juda sekin |
| **Orpheus 3B** | **~45.0** | ❌ Imkonsiz |

**MUHIM nuans:** bu 2015-yildagi 6-yadroli 35W Xeon. Sizda **16 yadro** bor va u ancha yangi — mening baholashim (manba emas): bu RTF'larni ~2–4 barobar yaxshilash mumkin, ya'ni Kokoro CPU'da ~0.7–1.4 RTF, XTTS ~1–2 RTF. Lekin bu faqat baholash.

Boshqa CPU hisobotlar:
- **Chatterbox Nano (110M):** "3x faster than realtime on **8 CPU cores**" — [rasmiy](https://github.com/resemble-ai/chatterbox)
- **Fish-Speech (RTX 5070, GPU'da!):** "~1:5 real-time factor (1 minute audio = 5 minutes processing)"; qisqa gaplar "30-60 seconds", o'rtacha matnlar "2-3 minutes" — [sneekes.app](https://sneekes.app/posts/fish_speech_installation_guide_for_rtx_5070_on_wsl2/). **Bu RTF ≈ 5** — juda sekin, GPU'da bo'lsa ham. Rasmiy "RTF 0.12" ([codesota](https://www.codesota.com/guides/tts-models)) bilan **jiddiy ziddiyat** — ehtimol birinchi ishga tushirish / noto'g'ri sozlash.
- **F5-TTS CPU:** "Works — the API falls back to `cpu` — but the README publishes no CPU speed figures" — [localaimaster](https://localaimaster.com/blog/f5-tts-setup-guide)
- **F5-TTS GPU tezligi (clore.ai):** RTX 3090, 100 belgi: "0.5s" (5x realtime); RTX 4090, 100 belgi: "0.3s" (8x); RTX 4090, 500 belgi: "1.2s" (10x) — [clore.ai](https://docs.clore.ai/guides/audio-and-voice/f5-tts)

---

## 6. 8 GB REALLIK: nimalar birga sig'adi

### 6.1. TTS-LAB'ning aniq hisob-kitoblari

**RTX 3060 12 GB (muvaffaqiyat):**
> "Orpheus bf16 (6.4GB) + XTTS (3.5GB) = **9.9GB ✅ both loaded**"

**RTX 4060 8 GB (muvaffaqiyatsizlik):**
> "Orpheus bf16 (6.4GB) = **OOM risk ❌**"

**RTX 5060 8 GB, FP4 bilan (muvaffaqiyat — lekin 3050'da FP4 YO'Q):**
> "Orpheus FP4: 0.9 GB + KV cache 0.3 GB = 1.2 GB... fits XTTS (2GB) + Chatterbox (1.5GB) + Kokoro (0.5GB)... **Total used: 6.2 GB of 8 GB ✅**"

**Bir vaqtda yuklanadigan engine soni:**
> RTX 3060 12 GB: 5–6 · RTX 4060 Ti 16 GB: 7–8 · **RTX 5060 8 GB: 4–5** · RTX 5070 Ti 16 GB: 7–8

— barchasi [TTS-LAB](https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md)

**Sizning 3050 8 GB uchun amaliy xulosa (mening baholashim, manba emas):** FP4/FP8 yo'qligi sababli 8 GB da bf16/fp16'da **bir vaqtda 2–3 ta kichik TTS engine** sig'adi (masalan XTTS 3.5 GB + Kokoro 0.5 GB + zahira), lekin Orpheus bf16 (6.4 GB) yoki VibeVoice-1.5B (6–7 GB) **yolg'iz o'zi** ishlashi kerak.

### 6.2. Windows desktop qancha VRAM oladi?

**Hujjatlashtirilgan raqam: ma'lumot topilmadi.** Tekshirgan manbalarimda bu raqam yo'q — [bestgpusforai 3060 vs 4060](https://www.bestgpusforai.com/gpu-comparison/3060-vs-4060) sahifasida ham: "The article contains **no explicit statements** about how much VRAM Windows consumes on these GPUs."

**Mening baholashim (MANBA EMAS, o'z xulosam):** Windows 11 WDDM desktop kompozitori + brauzer + IDE odatda 0.6–1.5 GB VRAM oladi. Ya'ni 8 GB kartada amalda **6.5–7.4 GB** qoladi. Buni kamaytirish yo'llari: brauzerda hardware acceleration'ni o'chirish, IDE'ni yopish, modellarni ishga tushirishda boshqa GPU ilovalarini to'xtatish.

Bilvosita dalil: VibeVoice uchun spheron "**Add 10% headroom** for production traffic" deydi — [spheron](https://www.spheron.network/tools/gpu-recommender/microsoft/VibeVoice-1.5B). 5.9 GB + 10% = 6.5 GB, desktop bilan birga 7.5–8 GB → chegarada.

### 6.3. 8 GB kartada hujjatlashtirilgan MUVAFFAQIYAT hisobotlari (iqtibos)

**XTTS fine-tune, RTX 3070 8 GB:**
> "I now on 3070 8Gb easilly finetuning. VRAM using show up to 17Gb in Task Manager."
> — [coqui-ai/TTS Disc #3268](https://github.com/coqui-ai/TTS/discussions/3268) (community report)

**8 GB GPU, drayver yangilashdan keyin:**
> "Another user experienced OOM errors even with BATCH_SIZE=1, but after driver update, '**completed 1 epoch**' successfully."
> — [coqui-ai/TTS Disc #3268](https://github.com/coqui-ai/TTS/discussions/3268)

**Higgs Audio, RTX 3050 6 GB (ComfyUI offload bilan):**
> "**RTX 3050 (6 GB): Functional via ComfyUI implementation**"
> "**6 GB VRAM + 44 GB RAM:** One user confirmed successful operation via ComfyUI"
> — [bosonai/higgs-tts-3-4b discussion #3](https://huggingface.co/bosonai/higgs-tts-3-4b/discussions/3) (community reports)
>
> **LEKIN** 44 GB RAM talab qiladi — sizda 15.9 GB. Bu yo'l sizga YOPIQ.

**8 GB uchun rasmiy "minimal GPU" tavsiyalari:**
- Chatterbox: "The minimum recommended GPU is **RTX 4060 (8 GB)**" — [localaimaster](https://localaimaster.com/blog/best-local-tts-models)
- F5-TTS: "**RTX 4060 (8 GB)** as minimum" — [localaimaster](https://localaimaster.com/blog/best-local-tts-models)
- XTTS-v2: "**RTX 4060 (8 GB)** as minimum" — [localaimaster](https://localaimaster.com/blog/best-local-tts-models)
- VibeVoice-1.5B: "an **8 GB consumer card** (e.g., RTX 3060) is generally sufficient for inference" — [localaimaster](https://localaimaster.com/blog/vibevoice-local-setup-guide)
- IndexTTS-2: "The Local Lab recommends a CUDA-compatible NVIDIA GPU with **at least 8 GB VRAM** for its Windows build" — [zimaspace](https://shop.zimaspace.com/pages/index-tts-nvidia-gpu-hardware-requirements)

---

## 7. TIZIM RAM (15.9 GB) — ogohlantirishlar

### 7.1. Hujjatlashtirilgan RAM talablari

| Model/vazifa | Minimal RAM | Tavsiya | ~14 GB dan oshadimi? | Manba |
|---|---|---|---|---|
| **XTTS fine-tune (8 GB VRAM yo'li)** | — | **24 GB+ MAJBURIY** | 🔴 **HA — OSHADI** | [AllTalk Wiki](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version)) |
| **XTTS fine-tune (umumiy)** | 16 GB "absolute minimum" | 24 GB+ | 🟠 Chegarada (sizda 15.9) | [AllTalk Wiki](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version)) |
| **GPT-SoVITS inference** | 16 GB | — | 🟠 Chegarada | [gpt-sovits.org](https://gpt-sovits.org/gpt-sovits-system-requirements-complete-setup/) |
| **GPT-SoVITS fine-tune** | — | **32 GB** ("model fine-tuning, audio dataset processing, long speech generation") | 🔴 **HA** | [gpt-sovits.org](https://gpt-sovits.org/gpt-sovits-system-requirements-complete-setup/) |
| **F5-TTS** | 16 GB | 32 GB (optimal 64 GB) | 🟠 Chegarada | [clore.ai](https://docs.clore.ai/guides/audio-and-voice/f5-tts) |
| **F5-TTS training (num_workers=16 default)** | — | — | 🔴 **HA** — `num_workers=1` ga tushirish majburiy | [Disc #57](https://github.com/SWivid/F5-TTS/discussions/57) |
| **Chatterbox** | — | 16 GB | 🟠 Chegarada | [clore.ai](https://docs.clore.ai/guides/audio-and-voice/chatterbox-tts) |
| **Higgs Audio (6 GB VRAM offload yo'li)** | — | **44 GB** (community) | 🔴 **HA — imkonsiz** | [HF disc #3](https://huggingface.co/bosonai/higgs-tts-3-4b/discussions/3) |
| Boshqalar (CosyVoice, IndexTTS, MaskGCT, VibeVoice, Zonos, Llasa, Spark-TTS, OpenVoice) | **ma'lumot topilmadi** | — | — | — |

### 7.2. Xulosa: RAM sizning eng tor joyingiz

1. **CUDA Sysmem Fallback yo'li (8 GB VRAM'da XTTS fine-tune) sizda ishonchli emas** — u 24 GB+ RAM talab qiladi, sizda 15.9 GB.
2. **F5-TTS training'da `num_workers` ni 1–2 ga tushirish majburiy** — aks holda 16 worker RAM'ni yemiraydi.
3. **Higgs/VibeVoice-Large'ning CPU offload yo'li yopiq** (40+ GB RAM kerak).
4. **Dataset preprocessing** (audio yuklash, VAD, Whisper transkripsiya) ham RAM yeydi — training bilan bir vaqtda qilmang, ketma-ket qiling.
5. **Amaliy tavsiya (mening xulosam):** Windows pagefile'ni kattalashtirish (32 GB+, SSD'da) va training paytida brauzer/IDE/OneDrive'ni yopish.

---

## 8. AGAR 8 GB YETMASA — eng arzon chiqish yo'li

Har bir vazifa uchun amaliy minimal VRAM (manbalar asosida):

| Vazifa | Amaliy minimal VRAM | Dalil |
|---|---|---|
| **Kichik TTS inference** (Kokoro, Piper, F5-TTS, XTTS, Chatterbox, GPT-SoVITS, Fish S1-mini) | **6–8 GB** | GPT-SoVITS ">=6G"; "RTX 4060 (8 GB) as minimum" |
| **O'rta TTS inference** (IndexTTS-2, VibeVoice-1.5B, Zonos, Orpheus 3B bf16) | **12 GB** | Orpheus bf16 6.4 GB 8 GB'da "OOM risk", 3060 12 GB'da ishlaydi |
| **Katta TTS inference** (Higgs Audio v2, VibeVoice-Large, Fish S2-Pro, Llasa-8B) | **16 GB minimal / 24 GB qulay** | "roughly 16 GB of VRAM as a practical minimum, and 24 GB for comfortable headroom"; Fish S2-Pro "24GB" |
| **GPT-SoVITS fine-tune** | **6 GB (batch=1)** | ✅ Rasmiy — sizda ishlaydi |
| **GPT-SoVITS DPO fine-tune** | **12 GB** | "A GPU with 12G VRAM or more" |
| **XTTS fine-tune (Windows + Sysmem)** | 8 GB VRAM + **24 GB RAM** | AllTalk rasmiy |
| **XTTS fine-tune (ishonchli)** | **12 GB (Win) / 16 GB (Linux)** | AllTalk rasmiy |
| **F5-TTS full fine-tune** | **12 GB da ishlagan** (community), **16 GB+ tavsiya**, 24 GB qulay | Disc #769 (4080 Laptop 12 GB), 3rd party |
| **Chatterbox GRPO** | **12 GB** | chatterbox-streaming |
| **VibeVoice-1.5B LoRA** | **16 GB** | VibeVoice-finetuning |
| **Chatterbox LoRA** | **18 GB** | chatterbox-streaming |
| **Llasa-1B LoRA 16-bit** | **16 GB (Colab T4 baseline)** | Unsloth notebooklar T4'da ishlaydi |
| **VibeVoice-7B LoRA** | **48 GB** | VibeVoice-finetuning |

**Eng arzon sakrash nuqtalari (mening tahlilim):**
- **12 GB** (RTX 3060 12 GB — ikkinchi qo'l eng arzon): XTTS fine-tune ishonchli, F5-TTS fine-tune mumkin, Orpheus bf16 + XTTS birga sig'adi ("9.9GB ✅ both loaded").
- **16 GB** (RTX 4060 Ti 16 GB / 5060 Ti 16 GB): VibeVoice LoRA, F5-TTS full fine-tune, Higgs inference (tor).
- **24 GB** (RTX 3090 — eng yaxshi narx/VRAM): hamma narsa, Chatterbox LoRA (18 GB) ham. F5-TTS community hisobotlarining aksariyati 3090/4090'da.
- **Cloud ijara** (sizning holatingizda eng oqilona): RTX 4090 24 GB `$0.14–0.42/hr` yoki RTX 3090 `$0.07–0.21/hr` ([clore.ai narxlari](https://docs.clore.ai/guides/audio-and-voice/chatterbox-tts)); spheron RTX 4090 24 GB `$0.58/hr` ([spheron](https://www.spheron.network/tools/gpu-recommender/microsoft/VibeVoice-1.5B)). Fine-tune 7 soat (F5-TTS, 4090, 93000 batch) → ~$1–4. **Yangi GPU sotib olishdan arzonroq.**

Cloud'ning ikkinchi foydasi: **Linux** → bitsandbytes 8-bit optimizer, DeepSpeed, flash-attn, pynini — hammasi ishlaydi. Bu Windows'da yo'qolgan barcha xotira-tejash vositalarini qaytaradi.

---

## 9. AMALIY TAVSIYALAR (sizning aniq konfiguratsiyangiz uchun)

### 9.1. Bugun 3050 8 GB + Windows'da ISHLATILADI (inference)

| Model | Ishonch | Izoh |
|---|---|---|
| **Kokoro-82M** | 🟢 Juda yuqori | ~0.5–1 GB. Lekin **zero-shot cloning YO'Q** — sizning maqsadingizga mos emas |
| **Piper** | 🟢 Juda yuqori | CPU, <100 MB. Cloning yo'q |
| **GPT-SoVITS** | 🟢 Yuqori | ≥6 GB, Windows one-click .7z, RTF 0.028 (4060Ti) — **ENG YAXSHI TANLOV** |
| **F5-TTS** | 🟢 Yuqori | ~2–4 GB. `--ref_text` ni har doim bering. 3050 6 GB'da 320 ms first-chunk |
| **XTTS-v2** | 🟢 Yuqori | ~2–6 GB. Sentence splitting'ni o'chirmang. 3050 **6 GB**'da OOM qayd etilgan, 8 GB'da ishlashi kerak |
| **Chatterbox / Turbo / Nano** | 🟡 O'rtacha | ~2–3 GB, lekin rasmiy Windows testi yo'q (Debian 11). 3050 6 GB'da 280 ms |
| **Fish-Speech / OpenAudio S1-mini** | 🟡 O'rtacha | 4–6 GB o'lchangan. Community WSL2 ishlatgan |
| **CosyVoice 2** | 🟡 O'rtacha | pynini uchun conda majburiy; VRAM raqamlari 0.6–8 GB oralig'ida ishonchsiz |
| **Spark-TTS** | 🟡 O'rtacha | VRAM ma'lumot topilmadi, lekin 0.5B. Windows guide bor |
| **Zonos** | 🟠 Tor | Rasmiy 6 GB+ min. Windows fork + CUDA 12.4 + VS2022 |
| **IndexTTS-2** | 🟠 Tor | ~6–7 GB + 5.9 GB model fayllari. DeepSpeed'ni `--all-extras` siz o'tkazib yuborish |
| **VibeVoice-1.5B** | 🟠 Tor | 6–7 GB → desktop bilan birga OOM xavfi |
| **Orpheus 3B** | 🔴 Past | bf16 6.4 GB = "OOM risk" 8 GB da; FP8/FP4 Ampere'da yo'q; vLLM/WSL |
| **Higgs Audio v2** | 🔴 Yo'q | "8 GB cards aren't enough"; offload 44 GB RAM talab qiladi |
| **VibeVoice-Large** | 🔴 Yo'q | ~20 GB |
| **Llasa-3B/8B** | 🔴 Past/Yo'q | VRAM ma'lumot topilmadi; xcodec2 + DeepSpeed |
| **MaskGCT** | ❓ Noma'lum | Hech qanday VRAM ma'lumoti topilmadi |
| **OpenVoice v2** | ❓ Noma'lum | VRAM ma'lumoti topilmadi |

### 9.2. Bugun 3050 8 GB + Windows'da FINE-TUNE qilinadi

1. **🟢 GPT-SoVITS** — yagona ishonchli yo'l. `batch_size=1`, fp16 yoqilgan, DPO'ni o'tkazib yuborish. One-click paket barcha Windows muammolarini (ffmpeg, VS2017, monotonic_align) hal qilgan.
2. **🟠 XTTS-v2** — faqat CUDA Sysmem Fallback Policy yoqilgan holda, `BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=252`. **Lekin 15.9 GB RAM AllTalk'ning 24 GB talabidan past** → beqaror, juda sekin. Sinab ko'rish arziydi, muvaffaqiyatga ishonch past.
3. **🔴 F5-TTS** — 8 GB uchun hujjatlashtirilgan config yo'q. Eng past hisobot 12 GB (4080 Laptop). LoRA fork bor lekin VRAM raqami yo'q. Sinash mumkin: `batch_size_per_gpu` ~800–1200 (frame), `max_samples=4`, `grad_accumulation_steps` yuqori, `num_workers=1`. **Mening baholashim, manba emas.**
4. **🔴 Chatterbox** (LoRA 18 GB / GRPO 12 GB), **VibeVoice** (16 GB), **Llasa** (ko'p-GPU), **Higgs** (ma'lumot yo'q) — hammasi 8 GB'dan tashqarida.
5. **❌ Spark-TTS** — training kodi hali chiqmagan.

### 9.3. Cloud GPU MAJBURIY bo'ladigan holatlar

- F5-TTS to'liq fine-tune (sifatli natija uchun 16–24 GB)
- Chatterbox LoRA (18 GB)
- VibeVoice LoRA (16 GB / 48 GB)
- Llasa 3B/8B har qanday training
- Higgs Audio v2 inference yoki training
- **Har qanday 8-bit optimizer / DeepSpeed / bitsandbytes training** — Windows'da printsipial ishlamaydi

---

## 10. Barcha manbalar

### Rasmiy repo / model kartalari
- https://github.com/RVC-Boss/GPT-SoVITS
- https://github.com/SWivid/F5-TTS
- https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/train/README.md
- https://github.com/FunAudioLLM/CosyVoice
- https://github.com/index-tts/index-tts
- https://github.com/resemble-ai/chatterbox
- https://github.com/SparkAudio/Spark-TTS
- https://github.com/canopyai/Orpheus-TTS
- https://github.com/myshell-ai/OpenVoice
- https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct
- https://github.com/zhenye234/LLaSA_training
- https://github.com/dao-ailab/flash-attention
- https://github.com/deepspeedai/DeepSpeed/pull/8542
- https://huggingface.co/hexgrad/Kokoro-82M
- https://huggingface.co/HKUSTAudio/Llasa-3B
- https://huggingface.co/amphion/MaskGCT
- https://huggingface.co/fishaudio/openaudio-s1-mini/blob/main/README.md
- https://huggingface.co/docs/huggingface_hub/guides/manage-cache
- https://speech.fish.audio/finetune/
- https://bootphon.github.io/phonemizer/install.html
- https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning
- https://arxiv.org/pdf/2607.21042 (Faster IndexTTS-2 streaming)

### GitHub issue / discussion (jamoatchilik hisobotlari)
- https://github.com/coqui-ai/TTS/discussions/3268 — XTTS fine-tune VRAM, 3070 8 GB, Sysmem Fallback
- https://github.com/coqui-ai/TTS/discussions/1973 — espeak backend not found
- https://github.com/SWivid/F5-TTS/discussions/57 — F5-TTS fine-tune configlar, num_workers
- https://github.com/SWivid/F5-TTS/discussions/769 — 12 GB fine-tune, 8-bit optimizer off
- https://github.com/SWivid/F5-TTS/issues/81 — RTF 3.0 (erta demo)
- https://github.com/index-tts/index-tts/issues/393 — DeepSpeed Windows LNK1181 aio.lib
- https://github.com/index-tts/index-tts/issues/395 — dskernels missing
- https://github.com/index-tts/index-tts/issues/392 — Windows install issue
- https://github.com/index-tts/index-tts/issues/61 — pynini conda yechimi (to'liq buyruqlar)
- https://github.com/boson-ai/higgs-audio/issues/29 — 12 GB OOM, expandable_segments
- https://huggingface.co/bosonai/higgs-tts-3-4b/discussions/3 — 16/24 GB, "8 GB aren't enough", 3050 ComfyUI
- https://github.com/SpenserCai/ComfyUI-FunAudioLLM/issues/7 — pynini/webrtcvad Windows
- https://github.com/2noise/ChatTTS/issues/162 — pynini conda-forge

### Jamoatchilik fork / Windows buildlar
- https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version)
- https://github.com/daswer123/xtts-finetune-webui
- https://github.com/davidbrowne17/chatterbox-streaming
- https://github.com/voicepowered-ai/VibeVoice-finetuning
- https://github.com/instavar/f5-tts-lora-finetuning
- https://github.com/kustomzone/Zonos-for-windows
- https://github.com/Foadsf/zonos-windows-tts
- https://github.com/faridnasiri/TTS-LAB/blob/main/docs/reference/GPU_QA_REFERENCE.md
- https://github.com/devnen/Chatterbox-TTS-Server/blob/main/documentation.md

### Uchinchi tomon / benchmark / kalkulyator (ehtiyotkorlik bilan)
- https://localaimaster.com/blog/voice-ai-vram-requirements-by-gpu
- https://localaimaster.com/blog/best-local-tts-models
- https://localaimaster.com/blog/f5-tts-setup-guide
- https://localaimaster.com/models/coqui-tts
- https://localaimaster.com/blog/vibevoice-local-setup-guide
- https://localaimaster.com/blog/gpt-sovits-voice-cloning-guide
- https://www.codesota.com/guides/tts-models
- https://docs.aihub.gg/tts/gpt-sovits/
- https://gpt-sovits.org/gpt-sovits-system-requirements-complete-setup/
- https://docs.clore.ai/guides/audio-and-voice/f5-tts
- https://docs.clore.ai/guides/audio-and-voice/chatterbox-tts
- https://www.spheron.network/blog/voice-ai-gpu-infrastructure/
- https://www.spheron.network/blog/deploy-open-source-tts-gpu-cloud-2026/
- https://www.spheron.network/tools/gpu-recommender/microsoft/VibeVoice-1.5B
- https://www.spheron.network/tools/gpu-recommender/bosonai/higgs-tts-v3-4b/
- https://www.runcomfy.com/comfyui-nodes/VibeVoice-ComfyUI
- https://localops.tech/model/cosyvoice-2-instruct
- https://www.madebyagents.com/models/cosyvoice-2-0
- https://www.madebyagents.com/models/cosyvoice-2-0/hardware
- https://nexgpu.net/en/models/fish-speech/
- https://sneekes.app/posts/fish_speech_installation_guide_for_rtx_5070_on_wsl2/
- https://shop.zimaspace.com/pages/index-tts-nvidia-gpu-hardware-requirements
- https://shop.zimaspace.com/pages/index-tts-hardware-requirements
- https://theneuralbase.com/bitsandbytes/learn/beginner/windows-support-caveats/
- https://www.mindfiretechnology.com/blog/archive/installing-bitsandbtyes-for-windows-so-that-you-can-do-peft/
- https://www.bestgpusforai.com/gpu-comparison/3060-vs-4060
- https://www.gpu-mart.com/blog/how-to-install-cosyvoice
- https://civitai.com/articles/9555/installing-triton-and-sage-attention-flash-attention-and-x-formers-win
- https://pasqualepillitteri.it/en/news/7051/gpt-sovits-voice-cloning-guide
- https://dev.to/czmilo/cosyvoice-2025-complete-guide-the-ultimate-multi-lingual-text-to-speech-solution-4l39
- https://gigagpu.com/xtts-v2-vram-requirements/ (403 — faqat qidiruv snippet'i)
- https://gigagpu.com/tts-latency-benchmarks/ (403 — faqat qidiruv snippet'i)

---

## 11. Metodologiya va cheklovlar

- **WebSearch budjeti:** 18/18 ishlatildi (to'liq tugadi). Qolgan ma'lumotlar WebFetch bilan to'plandi.
- **Olinmagan sahifalar:** `gigagpu.com` (HTTP 403 × 2), `deepwiki.com/fishaudio` (HTTP 429). Ularning ma'lumoti faqat qidiruv snippet'laridan olindi va shunday yorliqlandi.
- **`gh` CLI mavjud emas** bu muhitda, shuning uchun GitHub issue'larini API orqali ko'p miqdorda qidirish imkoni bo'lmadi.
- **"ma'lumot topilmadi"** yozilgan joylarda raqam taxmin qilinmadi.
- **Mening o'z baholashlarim** har doim "mening baholashim (manba emas)" deb yorliqlangan.
- **Ziddiyatlar** har doim ikki tomonli ko'rsatildi (XTTS 2 GB vs 6 GB; CosyVoice 0.6 GB vs 8 GB; F5-TTS RTF 0.147 vs 3.0; Fish-Speech RTF 0.12 vs ~5).

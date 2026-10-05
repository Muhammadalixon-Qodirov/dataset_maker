# Zero-shot ovoz klonlash: bir kishi uchun ENG ARZON HAQIQIY YO'L

**Tadqiqot sanasi:** 2026-10-01
**Savol:** Katta laboratoriyalar bu modellarni 27 000–95 000 soat ma'lumot va 4–8 ta A100 bilan o'qitgan. Yolg'iz dasturchi (Windows 11, bitta RTX 3050 8GB, cloud GPU ijaraga olish imkoniyati) uchun ishlaydigan zero-shot ovoz klonlash tizimiga eng arzon yo'l qanday?

**Bu hujjat faqat NARX va AMALIYLIK haqida.** Arxitektura ichki tuzilishi, modellar katalogi, o'zbek tili xususiyatlari va batafsil VRAM raqamlari — boshqa agentlar zimmasida.

**Manbalar belgisi:**
- 📄 — ilmiy maqola (peer-reviewed yoki arXiv preprint)
- 💬 — jamoa hisoboti (GitHub issue/discussion, HuggingFace model card, blog) — **maqola emas**
- 💰 — narx sahifasi (kuzatilgan sana ko'rsatilgan)
- 🧮 — **mening hisob-kitobim** (nashr etilgan raqam emas, arifmetikasi ko'rsatilgan)

---

## 0. Umumiy xulosa (bir qarashda)

| # | Yo'l | Kerakli ma'lumot | GPU | Vaqt | Taxminiy narx (USD) | Kutilayotgan sifat | Asosiy manba |
|---|------|------------------|-----|------|---------------------|---------------------|--------------|
| **A** | Tayyor zero-shot modelni o'zgartirmasdan ishlatish | 0 soat (faqat 3–10 sek referens) | RTX 3050 8GB (o'zingizdagi) | 0 | **$0** | Qo'llanadigan tilda yaxshi (XTTS SECS yuqori, F5 WER 2.42%); **qo'llanmaydigan tilda talaffuz buzuq** | 💬 [Coqui XTTS docs](https://docs.coqui.ai/en/latest/models/xtts.html), 📄 [F5-TTS](https://arxiv.org/html/2410.06885v3) |
| **B1** | **F5-TTS ni yangi tilga fine-tune** (char-level vocab) | 50h min, 100h yaxshi, 300h+ optimal (jamoa); 90h bilan muvaffaqiyat bor | 1× A100 80GB yoki 1× RTX 4090 | ~24 soat (A100, 90h data) / 2–5 kun (4090, 100–150k qadam) | **$16–$90** (bitta urinish) | Tushunarli, lekin talaffuz xatolari qoladi | 💬 [F5-TTS #1168](https://github.com/SWivid/F5-TTS/discussions/1168), 💬 [#143](https://github.com/SWivid/F5-TTS/discussions/143) |
| **B2** | **XTTS-v2 ni yangi tilga fine-tune** (tokenizer kengaytirish) | ~20h (DVAE'siz GPT fine-tune uchun yetarli); 100h+ "yaxshi ishlaydi" | 16GB+ VRAM (to'liq FT) | nashr etilmagan | 🧮 **$16–$52** (48–120 GPU-soat 4090da) | viXTTS/vnTTS mavjud, lekin raqamlar nashr etilmagan | 💬 [anhnh2002 repo](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages), 💬 [coqui #3992](https://github.com/coqui-ai/TTS/issues/3992) |
| **C1** | **SC-GlowTTS / YourTTS sinfi: speaker-encoder bilan noldan** | VCTK 44h / 98 spikr; 11 spikr bilan ham konvergensiya | 1× V100 (yoki 3090/4090) | 150k qadam (SC-GlowTTS) | 🧮 **$16–$90** | SECS 0.8046, MOS 3.78, Sim-MOS 3.999 (SC-GlowTTS); SECS 0.864, MOS 4.21 (YourTTS, 1200+ spikr bilan) | 📄 [SC-GlowTTS](https://ar5iv.labs.arxiv.org/html/2104.05557), 📄 [YourTTS](https://ar5iv.labs.arxiv.org/html/2112.02418) |
| **C2** | F5TTS_Small ni noldan | 945h (nashr etilgan konfiguratsiya) | 8× A100 80GB | 800k qadam | 🧮 **$139–$1 250** | WER 4.17%, SIM-o 0.54 (Seed-TTS test-zh) | 📄 [F5-TTS maqolasi](https://arxiv.org/html/2410.06885v3) |
| **D** | **LoRA / PEFT** (ovoz uchun, til uchun shubhali) | 2h (LJSpeech tajribasi) – 10h gacha | RTX 3090 Ti 24GB tasdiqlangan; CosyVoice3 LoRA **peak 6.96GB** | 500 update (F5) / 200 epoch (CosyVoice3) | **$2–$20** | F5 LoRA r32: WER 4.5012%, SIM 0.7398, UTMOS 4.3475 — to'liq FT dan yaxshi | 💬 [F5-TTS-LoRA](https://github.com/1109-christine/F5-TTS-LoRA), 💬 [cosyvoice3-lora](https://github.com/instavar/cosyvoice3-lora-finetuning) |

**Reyting (yolg'iz dasturchi uchun):** D (narx/samara) → B1 → B2 → C1 → A (baseline) → C2 (amalda mumkin emas).

Lekin **D yo'li yangi til qo'sha olmaydi** — pastda sababi.

---

## 1. YO'L A — Hech narsa o'qitmasdan, tayyor modelni ishlatish

Bu **mag'lub etilishi kerak bo'lgan baseline**. Halol baholash:

### Mavjud tayyor zero-shot modellar va ularning til qamrovi

| Model | O'qitish ma'lumoti | Qo'llanadigan tillar | Zero-shot sifat (nashr etilgan) |
|---|---|---|---|
| **XTTS-v2** | **27.3k soat, 16 til** (💬 [emergentmind](https://www.emergentmind.com/topics/xtts-model)); "clusters of NVIDIA A100 GPUs" | en, es, fr, de, it, pt, pl, **tr**, ru, nl, cs, ar, zh-cn, ja, hu, ko (+hi HF cardda) — 📄 [Coqui docs](https://docs.coqui.ai/en/latest/models/xtts.html) | FLORES+ 16 tilda o'rtacha **CER=2.06**; Bangla few-shot **MOS 4.601** (ground truth 4.809) — 💬 emergentmind |
| **F5-TTS Base** | **Emilia, 95K soat** (EN+ZH, filtrlangandan keyin); 335.8M parametr; **8× A100 80G, "over one week", 1.2M update** | faqat **EN + ZH** | LibriSpeech-PC test-clean: **WER 2.42%, SIM-o 0.66, RTF 0.31** (32 NFE) — 📄 [arXiv](https://arxiv.org/html/2410.06885v3) |
| **CosyVoice 2** | Tokenizer: **200 000 soat** (110 884h ZH + 99 918h EN). Model: ZH 130k h, EN 30k h, **JA 4.6k h, KO 2.2k h** | zh, en, ja, ko | 💬 [CosyVoice2 arXiv](https://arxiv.org/html/2412.10117v1) |
| **Fish-Speech 1** | **har biri 50k soat EN/ZH/JA = 150k soat** | en, zh, ja | 💬 [Fish-Speech blog](https://huggingface.co/blog/lengyue233/fish-speech-1) |
| **GPT-SoVITS v3** | **~7 000 soat**; 330M LLM + 77M SoVITS | — | 💬 [emergentmind](https://www.emergentmind.com/topics/cosyvoice2) |
| **IndexTTS-2.5** | nashr etilmagan (**ma'lumot topilmadi**) | zh, en, ja, es, ar | 💬 [index-tts repo](https://github.com/index-tts/index-tts) — **fine-tune retsepti umuman hujjatlashtirilmagan** |

### Halol baseline xulosasi

1. **Agar tilingiz ro'yxatda bo'lsa** — Yo'l A **$0 ga eng yaxshi tanlov**. XTTS-v2 3 sekundlik klip bilan klonlaydi (📄 Coqui docs), F5-TTS WER 2.42% beradi. Bundan yaxshiroq natijaga yolg'iz dasturchi real byudjetda erisha olmaydi.

2. **Agar tilingiz ro'yxatda bo'lmasa** — Yo'l A **buziladi**:
   - 💬 [coqui-ai/TTS Discussion #3457](https://github.com/coqui-ai/TTS/discussions/3457): "There are instances where the base model doesn't perform to its best on zero-shot inference with unseen data, and fine-tuning is still necessary for a subset of use-cases."
   - 💬 [ablt.dev Turkish TTS guide](https://www.ablt.dev/blog/turkish-tts/): Coqui'ning bir tilli VITS modellari "don't make pronunciation errors like XTTS, although the output speech quality won't be as good" — ya'ni XTTS ko'p tilli bo'lgani uchun talaffuz xatosi qiladi.

3. **Muhim struktural fakt:** 📄 [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) — Emilia-Large 215.6k soat, **6 til**: En 139.0k, Zh 50.3k, De 7.2k, Fr 8.8k, Ja 2.8k, Ko 7.5k. **Turkiy yoki Markaziy Osiyo tillari yo'q.** Demak F5-TTS'ning oldingi o'qitishi turkiy tillar uchun hech qanday akustik-lingvistik asos bermaydi.

4. **Qiziq nuqta (CosyVoice 2):** uning speech tokenizer faqat ZH/EN da o'qitilgan bo'lsa ham, **boshqa tillarda zero-shot qobiliyat ko'rsatdi** va JA/KO sintezi uchun ishlatildi (💬 [CosyVoice2](https://arxiv.org/html/2412.10117v1)). Bu B yo'li uchun yaxshi xabar: tokenizer qayta o'qitilishi shart emas.

5. **Litsenziya tuzoq:** XTTS-v2 — **CPML (nokommersiya)**; TurkicTTS — Apache 2.0; SpeechT5 — MIT (💬 ablt.dev). Agar mahsulot qilmoqchi bo'lsangiz, XTTS yo'li yuridik jihatdan tiqilib qoladi.

> **Baseline hukmi:** $0, 0 kun, 0 ma'lumot. Qo'llanadigan til uchun — mag'lub etish qiyin. Qo'llanmaydigan til uchun — talaffuz va intonatsiya buziladi, bu "ishlaydigan tizim" emas. **Shuning uchun B yo'li hal qiluvchi.**

---

## 2. YO'L B — Mavjud zero-shot modelni YANGI TILGA fine-tune qilish

Bu **eng muhim yo'l**. Quyida topilgan barcha konkret hisobotlar.

### 2.1 XTTS-v2 → yangi til

#### Rasmiy holat: hujjatlashtirilmagan

- 📄 [Coqui XTTS docs](https://docs.coqui.ai/en/latest/models/xtts.html): "Stay tuned as we continue to add support for more languages. If you have any language requests, please feel free to reach out." — **yangi til qo'shishning rasmiy protsedurasi yo'q.**
- 💬 [coqui-ai/TTS Issue #3992 "Finetune XTTS for new languages"](https://github.com/coqui-ai/TTS/issues/3992): Coqui maintainer'lar bu issue'ni **"wontfix"** deb belgilagan. Yagona javob: "feel free to help". Rasmiy yo'l yo'q.
- 💬 [coqui-ai/TTS Discussion #3689 "How to add a new language to an XTTS v2 model"](https://github.com/coqui-ai/TTS/discussions/3689): **thread javobsiz qolgan.** Foydalanuvchi `sanebg` bolgar tili uchun "48 hours... of royalty free audiobooks recorded in my studio. Female and male" va "2xRTX 4090" ega ekanini aytgan — **lekin natija hisoboti yo'q**. Keyinchalik thread'da F5-TTS ga o'tish tavsiya qilingan.

> ⚠️ **Birinchi ogohlantirish signali:** 48 soat studio sifatli ma'lumot + 2× RTX 4090 ega odam natija joylamagan. Bu "tried and silently gave up" alomati.

#### Jamoa protsedurasi (aslida ishlaydigan yo'l)

💬 [anhnh2002/XTTSv2-Finetuning-for-New-Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages) — bu yo'lning *de facto* standarti:

```bash
# 1. Pretrained checkpoint
python download_checkpoint.py --output_path checkpoints/

# 2. Tokenizer/vocab kengaytirish  (KALIT QADAM)
python extend_vocab_config.py --output_path=checkpoints/ \
    --metadata_path datasets/metadata_train.csv \
    --language vi --extended_vocab_size 2000

# 3. DVAE fine-tune (IXTIYORIY)
CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py \
    --output_path=checkpoints/ --num_epochs=5 --batch_size=512 --lr=5e-6

# 4. GPT fine-tune (ASOSIY)
#    num_epochs=5, batch_size=8, grad_accumulation=4  -> effektiv batch 32
```

Konkret raqamlar:
- **Vocab kengaytirish: `--extended_vocab_size 2000`** (Vyetnam tili uchun)
- **Ma'lumot: "If you have enough short texts in your datasets (about 20 hours), you do not need to finetune DVAE."** → **~20 soat** DVAE'siz GPT fine-tune uchun yetarli chegara.
- 💬 [coqui #3992](https://github.com/coqui-ai/TTS/issues/3992) da muallif "over 100 hours of audio" bilan "works well" deb yozgan.
- **HiFiGAN dekoderini fine-tune qilish natijani YOMONLASHTIRADI** — repo muallifi tavsiya qilmaydi. (Bu real "tried and failed" ma'lumoti.)
- **GPU vaqti / wall-clock: ma'lumot topilmadi.**

#### Yangi tilga qo'shilgan XTTS modellari (natijalar bilan)

| Model | Til | Ma'lumot | GPU | Qadam/vaqt | Sifat |
|---|---|---|---|---|---|
| 💬 [capleaf/viXTTS](https://huggingface.co/capleaf/viXTTS) | Vyetnam | **viVoice, 888k sample** (soat — **ma'lumot topilmadi**) | **ma'lumot topilmadi** | **ma'lumot topilmadi** | **ma'lumot topilmadi**. Usul: "fine-tuned from XTTS-v2.0.3 by expanding the tokenizer to Vietnamese". FPT University HCMC bitiruv ishi. |
| 💬 [anhnh2002/vnTTS](https://huggingface.co/anhnh2002/vnTTS) | Vyetnam | **ma'lumot topilmadi** | **ma'lumot topilmadi** | **ma'lumot topilmadi** | Model card faqat ishlatish ko'rsatmalari; o'qitish tafsilotlari nashr etilmagan |
| 💬 [XTTSv2-BY (Kaggle)](https://www.kaggle.com/datasets/wisekinder/xttsv2-by/data) | Belarus | sahifa ochilmadi — **ma'lumot topilmadi** | — | — | — |

> ⚠️ **Jiddiy muammo:** XTTS yangi til fine-tune'larining **birortasi ham** soat/GPU/qadam/sifat raqamlarini nashr etmagan. Modellar mavjud va ishlaydi, lekin **takrorlanadigan narx ma'lumoti yo'q**. Bu B2 yo'lini rejalashtirish uchun yomon.

#### XTTS fine-tune uchun apparat (bir ovoz uchun, til uchun emas)

- 💬 [gokhaneraslan/XTTS_V2-finetuning](https://github.com/gokhaneraslan/XTTS_V2-finetuning): "Full fine-tuning requires significant GPU memory (16GB+). LoRA reduces this requirement substantially." Tavsiya: **"at least 30 minutes of transcribed speech data"**. O'qitish davomiyligi: "several hours to days".
  - ⚠️ **MUHIM CHEKLOV:** bu repo'da ochiq yozilgan: "language support is limited to XTTS V2's existing language capabilities" → **bu LoRA implementatsiyasi yangi til QO'SHMAYDI.**
- 💬 [erew123/alltalk_tts](https://github.com/erew123/alltalk_tts) (XTTS finetune moduli): **"At least 2 minutes of good quality speech"**; "Preferably 12GB+ VRAM on Windows. Minimum 16GB VRAM on Linux"; 18GB disk (fine-tune vaqtida 20GB). Epoch soni va vaqt — nashr etilmagan.
- 📄 Coqui docs: Gradio "Easy Training" — Colab'da "up to 40 minutes" (lekin bu bitta ovoz uchun, yangi til uchun emas).

> ⚠️ **RTX 3050 8GB uchun:** XTTS to'liq fine-tune **mumkin emas** (16GB+ kerak). Faqat LoRA mumkin — lekin LoRA yangi til bermaydi.

### 2.2 F5-TTS → yangi til (ENG YAXSHI HUJJATLASHTIRILGAN YO'L)

#### Jamoa tomonidan aytilgan ma'lumot talabi

💬 [SWivid/F5-TTS Discussion #143](https://github.com/SWivid/F5-TTS/discussions/143) — bu eng foydali manba:

| Maqsad | Tavsiya etilgan soat |
|---|---|
| Bitta ovoz (single speaker) | **10–15 soat** |
| Yangi til, ko'p spikr — minimum | **50 soat** ("to start") |
| Yangi til — yaxshi model | **kamida 100 soat** |
| Yangi til — optimal | **kamida 300 soat yoki ko'proq** |
| EN/ZH da sinov uchun | 2–5 soat |

Shu thread'dagi boshqa raqamlar:
- "Spanish, Indian (Malayalam) with extend tokens, Hungarian" — **extend tokens bilan muvaffaqiyatli** tajribalar.
- **RTX 3080 (10GB VRAM):** koreys bir spikr modeli, **~12 soat ma'lumot, 300K qadam, batch size 1600** — ishladi.
- **T4:** batch size'ni kamaytirish shart bo'ldi (VRAM yetmadi).
- 💬 Boshqa thread'dan: yangi til uchun "**90+ hours** of high-quality audio", mono WAV **44kHz (BigVGAN)** yoki **24kHz (Vocos)**.

> ⚠️⚠️ **ENG MUHIM OGOHLANTIRISH (#143):** "Single-speaker limitations: Trained models struggled to mimic different voices when fine-tuned on only one speaker's data."
>
> **Ya'ni: bitta ovozda fine-tune qilsangiz, zero-shot klonlash QOBILIYATI YO'QOLADI.** Yangi til uchun zero-shot kerak bo'lsa — **ko'p spikrli ma'lumot majburiy**. Bu sizning loyihangiz uchun eng kritik fakt: 10 sekunddan klonlash uchun model ko'p spikrni ko'rgan bo'lishi shart.

#### Konkret fine-tune hisobotlari (soat / GPU / qadam / vaqt)

| Loyiha | Til | Soat | GPU | Qadam | Wall-clock | Manba |
|---|---|---|---|---|---|---|
| **Polyak referens keysi** | Polyak | **~90 soat** | **1× A100 80GB** | — | **~24 soat** | 💬 [#1168](https://github.com/SWivid/F5-TTS/discussions/1168) |
| Polyak (OP) | Polyak | 142 soat tayyorlangan | — | sinov: 2h data → 2 500 qadam (crash) | — | 💬 #1168. **500 qadamda tushunarsiz; 2 500 qadamda "intelligible" lekin ko'p xato** (native polyak baholovchi) |
| `healthyfat` tavsiyasi | umumiy | — | **RTX 4090** | **100 000–150 000 qadam** | **2–5 kun** | 💬 #1168. Dastlabki tushunarlilik **10 000–20 000 qadam**da |
| **lpscr** | Yunon + Ingliz | **40h yunon + 20h LibriTTS-R** | **1× RTX 3090**, `batch_size_per_gpu=1618` | `epochs=11`, `lr=1e-5`, `warmup=500` | **"in half a day"** (~12 soat) | 💬 [#57](https://github.com/SWivid/F5-TTS/discussions/57) |
| **jpgallegoar** | Ispan | 218h (yakuniy model) | **1× RTX 4090**, batch 4000 | 1 epoch = 93 000 batch | **6h 50m 53s / epoch** | 💬 #57 + [F5-Spanish](https://huggingface.co/jpgallegoar/F5-Spanish) |
| `jpgallegoar/F5-Spanish` | Ispan | **218 soat** (Voxpopuli + regional crowdsourced + TEDx Spanish) | nashr etilmagan | **"Training Steps: 1,200,000"** | — | 💬 model card. ⚠️ **Nizo:** base checkpoint nomi ham `model_1200000.pt` — bu raqam yangi qadamlar emas, balki base hisoblagichni takrorlashi mumkin. Noaniq. |
| **bensonbs** | Xitoy | **~33 soat** | — | `lr=7.5e-5`, loss ~0.75 | — | 💬 #57 — **MUVAFFAQIYATSIZ**, pastda |
| **acul3** | Indonez + Italyan + Ingliz | — | — | — | **~3 kun** | 💬 #57 |
| `VIZINTZOR/F5-TTS-THAI` | Tay | **~500 soat** (Common Voice ~160h + Porjai ~300h) | **ma'lumot topilmadi** | **1 000 000 qadam** | **ma'lumot topilmadi** | 💬 [model card](https://huggingface.co/VIZINTZOR/F5-TTS-THAI). Sifat: WER/CER yo'q; "reading long text or some words is still not accurate" |
| `Misha24-10/F5-TTS_RUSSIAN` | Rus | **4 000 soat** rus + jami **5 000+ soat** (rus+ingliz) | **ma'lumot topilmadi** | **ma'lumot topilmadi** | — | 💬 [model card](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN). Urg'u belgisi `+` qo'llanadi |
| `hotstone228/F5-TTS-Russian` | Rus | Common Voice 17.0 + SOVA RUDevices + Golos | **ma'lumot topilmadi** | **813 000 qadam** | — | 💬 [model card](https://huggingface.co/hotstone228/F5-TTS-Russian). ⚠️ Card'da "100k hours" yozilgan — bu base F5-TTS ning Emilia raqami bilan chalkashlik, rus ma'lumoti emas. **Nizoli manba.** |
| `aihpi/F5-TTS-German` | Nemis | Common Voice + Emilia_DE | **8× NVIDIA H100** | **ma'lumot topilmadi** | **ma'lumot topilmadi** | 💬 [model card](https://huggingface.co/aihpi/F5-TTS-German) |
| `alien79/F5-TTS-italian` | Italyan | — | **8× RTX 4090** | — | **73+ soat** | 💬 qidiruv natijasidan olindi, model card to'g'ridan-to'g'ri tasdiqlanmadi — **ishonchlilik past** |
| `sarpba/F5-TTS-Hun` | Venger | **ma'lumot topilmadi** | **ma'lumot topilmadi** | **ma'lumot topilmadi** | — | 💬 [model card](https://huggingface.co/sarpba/F5-TTS-Hun). Qiziq: muallif "normalization layer that corrects many of the model's errors through input text adjustment" qo'shgan |

#### Mavjud F5-TTS til fine-tune'lari (to'liq ro'yxat, HF)

💬 [HuggingFace search](https://huggingface.co/models?search=F5-TTS): `Misha24-10/F5-TTS_RUSSIAN`, `hotstone228/F5-TTS-Russian`, `VIZINTZOR/F5-TTS-THAI`, `Maxdorger29/f5-tts-hungarian`, `sarpba/F5-TTS-Hun`, `mp3pintyo/F5-TTS-Hun`, `toandev/F5-TTS-Vietnamese`, `marduk-ra/F5-TTS-German`, `aihpi/F5-TTS-German`, **`marduk-ra/F5-TTS-Turkish`**, `ModelsLab/F5-tts-brazilian`, `traderpedroso/F5-TTS-BRAZILIAN-PORTUGUESE`, `PetrosStav/F5-TTS-Greek`, `andromeda0302/E2-F5-TTS-SPANISH`, `Eempostor/F5-TTS-INDO-FINETUNE`, `gentleman101/bhojpuri-f5-tts`.

> **Xulosa:** F5-TTS ga **kamida 12 yangi til** jamoa tomonidan qo'shilgan. Bu yo'l **amalda ishlaydi** — eng ko'p tasdiqlangan B yo'li. Turkiy til (turk) ham bor.

#### Rasmiy F5-TTS fine-tune vositalari

💬 [F5-TTS train README](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/train/README.md):
- `accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml`
- `finetune_gradio.py` — web UI (hujjatda mavjud)
- `++datasets.batch_size_per_gpu=19200` — CLI orqali batch override
- Dataset skriptlar: Emilia, Wenetspeech4TTS, LibriTTS, LJSpeech + custom CSV (`audio_file|text`)
- **Muhim ko'rsatma:** "The `use_ema = True` might be harmful for early-stage finetuned checkpoints" → kam update bilan fine-tune qilganda EMA'ni o'chirish kerak
- ⚠️ "The document doesn't specify model sizes, minimum data requirements, or recommended training step counts" — **rasmiy ma'lumot talabi hujjati yo'q**, hammasi jamoadan

### 2.3 CosyVoice 2 / Fish-Speech / GPT-SoVITS / IndexTTS2 → yangi til

| Model | Yangi til fine-tune hujjati |
|---|---|
| **CosyVoice 2** | Rasmiy yangi-til retsepti **ma'lumot topilmadi**. Lekin 📄 [maqola](https://arxiv.org/html/2412.10117v1) ko'rsatadi: ZH/EN da o'qitilgan tokenizer **boshqa tillarda zero-shot ishlaydi** va JA (4.6k h) / KO (2.2k h) fine-tune'ga qo'shilgan. Bu "tokenizer qayta o'qitish shart emas" degan kuchli dalil. LoRA repo mavjud (pastda, D yo'li). |
| **Fish-Speech** | 💬 [speech.fish.audio/finetune](https://speech.fish.audio/finetune/) — **minimal dataset soati, VRAM, LoRA vs full FT, qadam soni: hammasi hujjatda yo'q (ma'lumot topilmadi)**. Yangi til masalasi umuman ko'rib chiqilmagan. Ogohlantirish: "We highly do not recommend users to do fine-tuning on an RL trained model. Fine-tuning a model after RL can shift the model distribution, which may lead to degraded performance." |
| **GPT-SoVITS** | Rasmiy yangi-til retsepti **ma'lumot topilmadi**. Jamoa issue'lari bor: 💬 [#2595 koreys](https://github.com/RVC-Boss/GPT-SoVITS/issues/2595) ("loss is strange"), 💬 [#241 yapon](https://github.com/RVC-Boss/GPT-SoVITS/issues/241). ~7 000 soat base. 77M SoVITS moduli juda kichik → CSP-FT uchun qulay (D yo'li). |
| **IndexTTS / IndexTTS2** | 💬 [repo](https://github.com/index-tts/index-tts): **fine-tune protsedurasi umuman hujjatlashtirilmagan**, o'qitish soatlari nashr etilmagan, VRAM raqami yo'q. **Yolg'iz dasturchi uchun yopiq yo'l.** |
| **Chatterbox** | 💬 [gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning) — **yangi til uchun maxsus qurilgan**: "The setup script performs a 'Vocab Extension': it intelligently adds all unique characters from our 23-language grapheme set to the GPT-2 vocabulary." `new_vocab_size` ni `tokenizer.json` bilan moslashtirish shart. **Minimum 30+ daqiqa.** Qoida: "LoRA... Best for: Datasets that are 10 hours or less"; to'liq FT faqat ">10 hours" uchun. LoRA "~60% less VRAM". Natija raqamlari **nashr etilmagan**. |

### 2.4 Llasa / Spark-TTS (LLM-asosli) → LLM pretraining kam resursli tilga yordam beradimi?

- 💬 [HKUSTAudio/Llasa-3B](https://huggingface.co/HKUSTAudio/Llasa-3B) — LLaMA asosidagi TTS, bitta Transformer, autoregressiv next-token.
- 💬 [Llasagna blog (italyan fine-tune)](https://huggingface.co/blog/Steveeeeeeen/llasagna):
  - Ma'lumot: **CML-TTS italyan subset** (soat — **ma'lumot topilmadi**)
  - **"A minimum of 24 GB of VRAM is recommended for fine-tuning smaller models (e.g., 1B parameters)."** → RTX 3050 8GB **yetmaydi**
  - Usul: **LoRA emas, to'liq fine-tune** + `AutoLigerKernelForCausalLM` + 8-bit Adam
  - GPU turi/soni, vaqt, narx: **ma'lumot topilmadi**
  - Sifat metrikasi: **ma'lumot topilmadi** (faqat demo havolalari)
  - Qiyoslash uchun: nemis variantida (`Kartoffel`) muallif `SebastianBodza` izohda **"In total 2.5 million entries with around 7k hours"** ishlatganini aytgan → **7 000 soat**. Bu yolg'iz dasturchi uchun yetib bo'lmaydigan raqam.
- **Spark-TTS:** 💬 [SparkAudio/Spark-TTS-0.5B discussions/1](https://huggingface.co/SparkAudio/Spark-TTS-0.5B/discussions/1) — foydalanuvchi "is there any documentation for finetune model for extra language?" deb so'ragan, **maintainer'lardan javob yo'q**. Gruzin tili so'rovi ham javobsiz. **Yangi til hujjati mavjud emas.**
- **LLM pretraining yordam beradimi?** 📄 [Speech LLMs in Low-Resource Scenarios (arXiv 2508.05149)](https://arxiv.org/pdf/2508.05149): "Pretraining a speech-to-LLM projector on high-resource languages significantly boosts performance in low-resource settings, especially with limited fine-tuning data." → **Ha, printsipial jihatdan yordam beradi.** Lekin bu ASR/understanding uchun; TTS uchun to'g'ridan-to'g'ri o'lchov **ma'lumot topilmadi**.
- 📄 [Exploring Pre-training Benefits on Phoneme Addition (arXiv 2606.19792)](https://arxiv.org/pdf/2606.19792): "The target dataset often contains new phonemes unseen in pre-training, requiring the model to expand its phoneme inventory... during fine-tuning" — bu muammo "phoneme addition" deb nomlangan va maxsus tadqiqot mavzusi.

- 💬 [Unsloth TTS fine-tuning guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning): Orpheus-TTS (3B), Sesame-CSM (1B), **Spark-TTS (0.5B)**, Llasa-TTS (1B), Oute-TTS (1B) qo'llab-quvvatlanadi. "Unsloth supports any `transformers` compatible TTS model." LoRA 16-bit yoki to'liq FT. Namuna dataset: **Elise, ~3 soat, ~1 200 sample**. **Colab T4 da "1-2 hours"**. ⚠️ **"No guidance provided — the documentation does not address training for new languages specifically."** VRAM raqamlari ham berilmagan.

> **B yo'li umumiy hukmi:** **F5-TTS = eng yaxshi tanlov.** Eng ko'p mustaqil muvaffaqiyat (12+ til), eng ochiq jamoa raqamlari, eng kichik model (335.8M), eng past VRAM talabi, eng arzon tasdiqlangan run (**3090 da yarim kun**, lpscr). XTTS-v2 ikkinchi (viXTTS ishlaydi) lekin raqamlar yopiq, litsenziya nokommersiya va 16GB+ kerak.

---

## 3. YO'L C — KICHIK zero-shot modelni noldan o'qitish

### 3.1 YourTTS — klassik ARZON zero-shot model

📄 [YourTTS, arXiv 2112.02418](https://ar5iv.labs.arxiv.org/html/2112.02418):

**O'qitish ma'lumoti (jami ~1 380 soat, ammo har tilda juda kam):**
| Til | Dataset | Soat | Spikr |
|---|---|---|---|
| Ingliz | **VCTK** | **44 soat** | **109** |
| Ingliz | LibriTTS train-clean-100 + train-clean-360 | — | **~1 151 qo'shimcha spikr** |
| Portugal | TTS-Portuguese Corpus | **~10 soat** | **1** (bitta spikr!) |
| Fransuz | M-AILABS fr_FR | **104h (2 ayol) + 71h (3 erkak) = 175 soat** | **5** |

**Apparat va vaqt:**
- **GPU: bitta NVIDIA TESLA V100 32GB**
- **Batch size: 64**
- Qadamlar: Eksperiment 1 — LJSpeech'da 1M qadam, keyin VCTK'da 200K qadam; Eksperiment 2–3 — har biri **~140K qadam**; barcha eksperimentlarga Speaker Consistency Loss bilan **+50K qadam**
- **Wall-clock vaqt: maqolada ma'lumot topilmadi**

**Zero-shot natijalar (VCTK):**
- **SECS 0.864, MOS 4.21±0.04, Sim-MOS 4.16±0.05**

**Spikr adaptatsiyasi (1 daqiqadan kam ma'lumot bilan):**
- Ingliz erkak (**61 sek**): SECS **0.880**, Sim-MOS **4.08±0.09**
- Portugal ayol (**20 sek**): SECS **0.930**, Sim-MOS **4.43±0.06**
- Maqola xulosasi: **"more than 45 seconds of speech are more adequate to allow higher quality"**

> 🧮 **Mening hisobim (nashr etilgan emas):** VITS-sinfidagi model, batch 64, bitta V100 — tipik o'tkazuvchanlik ~1–2 qadam/sek. Jami ~1.39M qadam → 1.39M / 1.5 qadam/sek ≈ **257 soat ≈ 11 kun** bitta V100 da.
> - Lambda V100: **$0.79/soat** × 257 = **~$203**
> - RunPod 3090 Community **$0.22/soat** × 257 = **~$57** (3090 ≈ V100 tezligi)
> - Vast.ai 3090 **$0.39/soat** × 257 = **~$100**
> - RTX 3050 8GB da (batch kamaytirilgan holda): 🧮 taxminan **3–6 hafta**, lekin **$0**
>
> ⚠️ Bu mening ekstrapolyatsiyam, maqolada wall-clock yo'q. Faqat tartib-kattalik sifatida ishlatilsin.

**KALIT INSIGHT:** YourTTS **1 151+ spikr** ko'rgani uchun zero-shot ishlaydi — soat emas, **SPIKR SONI** muhim. Portugal tili **bitta spikr, 10 soat** bilan qo'shilgan va baribir zero-shot ishlagan (SECS 0.930 adaptatsiya bilan). **Bu kam resursli til uchun eng kuchli dalil.**

### 3.2 SC-GlowTTS — eng arzon nashr etilgan zero-shot model

📄 [SC-GlowTTS, arXiv 2104.05557](https://ar5iv.labs.arxiv.org/html/2104.05557) (YourTTS mualliflarining oldingi ishi):

- **Dataset: VCTK — 44 soat, 109 spikr, 48kHz.** Asosiy eksperimentlarda **~98 spikr** o'qitishda.
- **"Our model converges using only 11 speakers"** — 11 spikr bilan ham konvergensiya (5-bo'lim, kam resursli stsenariy)
- **GPU: "The training was carried out on an NVIDIA V100 GPU"** — **bitta V100**
- **Batch 128, lr 1e-3, 150K qadam** (Tacotron 2 uchun 210K qadam)
- **Natijalar (SC-GlowTTS-Trans + HiFi-GAN-FT):** **SECS 0.8046, MOS 3.78±0.07, Sim-MOS 3.999±0.07**
- Parametr soni: maqolada ko'rsatilmagan

> 🧮 **Mening hisobim:** 150K qadam, batch 128, flow-based model, bitta V100 → taxminan **40–70 soat**.
> - RunPod 3090 Community $0.22/h × 55h ≈ **$12**
> - Vast.ai 3090 $0.39/h × 55h ≈ **$21**
> - Lambda V100 $0.79/h × 55h ≈ **$43**
>
> **Bu eng arzon nashr etilgan zero-shot retsepti: ~$12–$43 va 44 soat ma'lumot.**

### 3.3 Eng kam ma'lumot bilan zero-shot: 5 DAQIQA

📄 [Low-Resource Multilingual and Zero-Shot Multispeaker TTS, AACL 2022](https://aclanthology.org/2022.aacl-main.56.pdf):
- Arxitektura: **Tacotron2-asosli**, tashqi speaker embedding bilan
- **"as little as 5 minutes of single speaker data"** per til bilan sintez sifati erishildi
- Natija: "zero-shot multispeaker TTS can perform comparably to single speaker models trained on 29 and 21 hours of data, and the ability to perform zero-shot multispeaker TTS is shared across languages, even those which have seen only 5 minutes of single speaker data"
- ⚠️ PDF matn ekstraksiyasi qisman ishladi — **aniq MOS raqamlari, GPU turi va o'qitish vaqti ma'lumot topilmadi**

> **Bu Yo'l C uchun eng kuchli ilmiy asos:** ko'p tilli baza + speaker encoder bo'lsa, yangi tilga **5 daqiqa** qo'shib ham zero-shot qobiliyat **o'tadi**. Ya'ni "yangi tilga 100 soat kerak" degan F5-TTS jamoa raqami **arxitekturaga bog'liq** — speaker-encoder dizaynlari codec-LM dizaynlaridan ancha arzon.

### 3.4 Boshqa kam resursli zero-shot ishlar

- 💬/📄 **Zero-Voice** ([techrxiv](https://www.techrxiv.org/users/882196/articles/1260796-zero-voice-a-low-resource-approach-for-zero-shot-text-to-speech)): qidiruv natijasiga ko'ra "**5-8 hours of transcribed data**" bilan kam spikrli datasetda zero-shot. Novel "Source Filter Network" prozodiyani unsupervised ajratadi. ⚠️ **To'liq matnni olish imkoni bo'lmadi (HTTP 403)** — faqat qidiruv tavsifi. **Tekshirilmagan preprint, soat/GPU/natijalar tasdiqlanmagan.**
- 📄 [kNN Retrieval for Simple and Effective Zero-Shot Multi-speaker TTS (arXiv 2408.10771)](https://arxiv.org/pdf/2408.10771) — retrieval-asosli, o'qitishsiz yondashuv. Tafsilotlar o'qilmadi.
- 📄 [BnTTS: Few-Shot Speaker Adaptation in Low-Resource (NAACL 2025 Findings)](https://aclanthology.org/2025.findings-naacl.279.pdf) — bengal tili uchun zero-shot modelni moslashtirish. ⚠️ **PDF o'qilmadi (binary) — raqamlar ma'lumot topilmadi.** Lekin 💬 emergentmind XTTS maqolasidan: "**MOS in Bangla attains 4.601 for few-shot adaptation**, approaching ground truth (4.809)" — bu kam resursli tilda XTTS adaptatsiyasi ishlaganini ko'rsatadi.
- 📄 [ESPnet2-TTS (arXiv 2110.07840)](https://ar5iv.labs.arxiv.org/html/2110.07840): "ESPnet2-TTS supports... zero-shot adaptation with pre-trained speaker embeddings". 📄 [ESPnet-SPK (arXiv 2401.17230)](https://arxiv.org/html/2401.17230v2): "currently supports over 30 recipes... including multi-speaker TTS". → **ESPnet'da tayyor zero-shot retsept bor.**
- **Amphion:** zero-shot multi-speaker + speaker-encoder retsepti bo'yicha **konkret ma'lumot topilmadi** (qidiruvda Amphion retsepti chiqmadi; faqat Emilia dataseti egasi sifatida).
- **Matcha-TTS / VITS + tashqi speaker encoder (DIY):** maxsus "zero-shot Matcha-TTS" nashr etilgan ishi **ma'lumot topilmadi**. Lekin SC-GlowTTS/YourTTS aynan shu dizayn (flow/VITS + H/ASV speaker encoder) va ular ishlaydi — ya'ni DIY dizayn ilmiy jihatdan asoslangan, faqat Matcha variantida nashr yo'q.

### 3.5 F5-TTS'ning o'z KICHIK konfiguratsiyasi

📄 [F5-TTS maqolasi](https://arxiv.org/html/2410.06885v3) + 💬 repo config'lari:

**Nashr etilgan konfiguratsiyalar:** `E2TTS_Base.yaml`, `E2TTS_Small.yaml`, `F5TTS_Base.yaml`, `F5TTS_Small.yaml`, `F5TTS_v1_Base.yaml`, `F5TTS_v1_Small.yaml` ([configs/](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/configs))

| | **F5-TTS Base** | **F5-TTS Small** |
|---|---|---|
| Parametr | **335.8M** | **~155M** (ba'zi joyda 158M) |
| Arxitektura | 22 DiT qatlam, 16 head, 1024/2048 dim; 4 ConvNeXt V2 qatlam, 512/1024 | `dim=768, depth=18, heads=12` (💬 `F5TTS_Small.yaml`) |
| Dataset | **Emilia, 95K soat** (EN+ZH) | **WenetSpeech4TTS Premium, 945 soat** (mandarin) |
| GPU | **8× NVIDIA A100 80G** | nashr etilmagan |
| Update | **1.2M** | **800K** |
| Batch | **307 200 audio frame (0.91 soat)** | maqolada: base'ning **yarmi**. ⚠️ **NIZO:** `F5TTS_Small.yaml` da `38400/GPU × 8 GPU = 307 200` — ya'ni shipped config base bilan **bir xil** batch ishlatadi. Maqola va config qarama-qarshi. |
| lr | 7.5e-5, 20K warmup | 7.5e-5, `epochs=11` |
| Vaqt | **"over one week"** | **ma'lumot topilmadi** |
| Natija | LibriSpeech-PC test-clean: **WER 2.42%, SIM-o 0.66** | Seed-TTS test-zh: **WER 4.17%, SIM-o 0.54** |

**ENG MUHIM FAKT:** **F5-TTS Small 945 soatda o'qitilgan va zero-shot ishlaydi** (SIM-o 0.54). Bu savolga ("1 000 soatdan kam bilan nashr etilgan zero-shot model bormi?") **ha** degan javob — **945 soat**, rasmiy maqolada, qayta ishlab chiqarilishi mumkin konfiguratsiya bilan.

> 🧮 **Noldan o'qitish narxi (mening hisobim):**
>
> Base: 8 GPU × 7 kun × 24 soat = **1 344 A100-soat** minimum ("over one week" → kamida).
>
> Small uchun masshtablash (compute ∝ parametr × batch × update):
> - **Optimistik:** 1 344 × (155/335.8) × 0.5 × (800/1200) = 1 344 × 0.462 × 0.5 × 0.667 ≈ **207 A100-soat**
> - **Konservativ** (parametr bo'yicha masshtablamasdan, attention/overhead chiziqli emasligi uchun): 1 344 × 0.5 × 0.667 ≈ **448 A100-soat**
> - **Eng konservativ** (shipped config to'liq batch ishlatsa): 1 344 × (800/1200) ≈ **896 A100-soat**
>
> Narxi (2026-10-01 kuzatilgan):
> | Provayder | $/A100-soat | 207h | 448h | 896h |
> |---|---|---|---|---|
> | Vast.ai A100 SXM4 80GB | **$0.67** | **$139** | **$300** | **$600** |
> | RunPod A100 80GB PCIe Community | **$1.19** | **$246** | **$533** | **$1 066** |
> | RunPod A100 80GB Secure | **$1.59** | **$329** | **$712** | **$1 425** |
> | Lambda A100 SXM 80GB (8x) | **$2.79** | **$577** | **$1 250** | **$2 500** |
>
> **Diapazon: $139 – $2 500.** Bitta urinish uchun. Haqiqatda 2–3 urinish kerak bo'ladi (hyperparameter, dataset xatolari) → **$400 – $7 500**.

> ⚠️ **Lekin bu C2 yo'lining asosiy muammosi emas.** Asosiy muammo: **945 SOAT MA'LUMOT**. Yolg'iz dasturchi uchun GPU $300 muammo emas; 945 soat tozalangan, transkripsiyalangan, ko'p spikrli o'zbek audio — **mana asosiy devor**. Shuning uchun C2 **amalda mumkin emas**.

---

## 4. YO'L D — LoRA / PEFT / adapter fine-tuning

### 4.1 F5-TTS LoRA — nashr etilgan eng aniq taqqoslash

💬 [1109-christine/F5-TTS-LoRA](https://github.com/1109-christine/F5-TTS-LoRA):

**Setup:** LJSpeech, **2 soat** audio, **500 update**, bitta seed (666), 100 held-out utterance test seti, torch 2.4.1.

| Usul | WER (%) ↓ | Spikr o'xshashligi ↑ | UTMOS ↑ |
|---|---|---|---|
| Official Base | 4.6837 | 0.6470 | 4.2590 |
| **To'liq fine-tune** | 4.7445 | **0.7597** | 4.3046 |
| **LoRA r8** | 4.5620 | 0.7266 | 4.3318 |
| **LoRA r16** | 4.5620 | 0.7289 | **4.3589** |
| **LoRA r32** | **4.5012** | 0.7398 | 4.3475 |

> **LoRA r32 WER va UTMOS bo'yicha to'liq fine-tune'dan YAXSHI.** Faqat spikr o'xshashligida to'liq FT oldinda (0.7597 vs 0.7398).
>
> Mualliflar ogohlantirishi: **"These are single-seed observations, not claims of statistical significance or a universally best rank."**
>
> ⚠️ GPU modeli, VRAM, o'qitish vaqti — **ma'lumot topilmadi**.

💬 [instavar/f5-tts-lora-finetuning](https://github.com/instavar/f5-tts-lora-finetuning):
- Default **rank 16, alpha 16**
- Trainable: **~0.9% parametr** → **2 883 584 / 339 980 388 parametr**
- Adapter hajmi: **~6 MB (rank 16)** / 12 MB, to'liq FT **~2.7 GB** ga nisbatan
- **RTX 3090 Ti 24GB da tasdiqlangan**, **10 850 training row**
- Inference: RTF 0.0394 (L20 GPU, concurrency 2)
- ⚠️ **O'qitish vaqti va peak VRAM — ma'lumot topilmadi**
- ⚠️ **Yangi til haqida:** "The repository targets 'new voice' adaptation through LoRA. The documentation does not explicitly address whether LoRA works for entirely new languages versus just speaker adaptation."

### 4.2 CosyVoice3 LoRA — ENG QIMMATLI MUVAFFAQIYATSIZLIK HISOBOTI

💬 [instavar/cosyvoice3-lora-finetuning](https://github.com/instavar/cosyvoice3-lora-finetuning) — "corrected from failed full-SFT attempt"

**MUVAFFAQIYATSIZ to'liq SFT:**
- Dataset: IMDA NSC FEMALE_01 (**16 535 train / 870 dev** utterance)
- GPU: **RTX 3090 Ti (24 GB)**
- **"All 506M parameters were trained via upstream `cosyvoice/bin/train.py`"**
- **"Massive overfitting after epoch 1"** — CV loss darhol divergensiya qildi
- **"At epochs 8-10, generating >10 second audio required 11-18 seed attempts"**
- **"12+ evaluation directories tried different epochs... None produced production quality"**
- Eng yaxshi CV loss: **2.900 (faqat epoch 1)**
- Checkpoint: **~4 GB har biri, jami 174 GB** disk
- O'qitish **epoch 41** gacha ketdi, gating yo'q edi

**MUVAFFAQIYATLI LoRA (bir xil dataset):**
- **200 epoch**, eng yaxshi checkpoint **epoch 12**, CV loss **3.044**
- Trainable: **2.16M (0.44%)**
- Checkpoint: **8.3 MB**
- **Peak VRAM: 6.96 GB** (to'liq SFT'da **13.08 GB** — **47% kamayish**)
- Tezlik: **6.9 sample/sek (1.8× tezroq)**
- Gradient norm stabil (**1.4–4.0**), to'liq SFT'da "explosive"
- **"LoRA showed 12x more useful training before overfitting vs full SFT's immediate divergence"**

**9 ta tuzoq (to'liq ro'yxat):**
1. Bitta spikr uchun **to'liq SFT'dan voz kechish** — LoRA ishlatish
2. **Training loss'ga ishonmaslik** — checkpoint tanlash uchun validation loss
3. **Early stopping yoqilishi**
4. **Checkpoint hajmi usulni ko'rsatadi** — GB hajmli fayl = tasodifan to'liq SFT
5. **PEFT wrapping `embed_tokens` ga kirishni buzadi** — proxy patching kerak
6. **Parquet'dagi embedding dtype** `numpy.float32` bo'lishi shart, Python list emas
7. **Prompt formatlash sezgir** — `<|endofprompt|>` joylashuvi izchil bo'lsin
8. **Qwen pretrain yo'li `CosyVoice-BlankEN`**, generic Qwen emas
9. **Upstream `max_epoch` konfiglarni bosib ketadi** — YAML yoki CLI override kerak

> 🔥 **Bu sizning loyihangiz uchun eng qimmatli hujjat.** Bitta 24GB GPU'da, bitta spikrda, to'liq SFT 506M parametrni o'qitib **ishlab chiqarish sifatiga yeta olmadi**; LoRA 2.16M parametr bilan **6.96 GB VRAM** da ishladi. **RTX 3050 8GB ham 6.96 GB ga yaqin** — ya'ni sizning kartangiz LoRA uchun chegarada.

### 4.3 Qismiy fine-tune (CSP-FT) — nashr etilgan maqola

📄 [Characteristic-Specific Partial Fine-Tuning, arXiv 2501.14273](https://arxiv.org/html/2501.14273v1):
- Base modellar: **GPT-SoVITS, VALLE-X, CosyVoice**
- **~8% parametr** fine-tune qilinadi → GPT-SoVITS'da **6.3M / 77.46M**
- Tahlil fazasi: **244 soat** (11 emotsional speech dataset)
- **GPU: NVIDIA RTX 3090**
- **~2× tezroq**: 1.91× (GPT-SoVITS), 2.62× (VALLE-X), 2.12× (CosyVoice)
- Sifat: **"comparable to, or even surpassing, full fine-tuning"**
- **WER: CSP-FT 8.49% vs to'liq FT 10.25%** (GPT-SoVITS) → **catastrophic forgetting kamaydi**

### 4.4 XTTS LoRA

💬 [gokhaneraslan/XTTS_V2-finetuning](https://github.com/gokhaneraslan/XTTS_V2-finetuning):
- LoRA: `target_modules: ["c_attn", "c_proj", "c_fc"]`, `r: 8`, `lora_alpha: 32`
- "Full fine-tuning requires significant GPU memory (16GB+). **LoRA reduces this requirement substantially.**" (LoRA uchun aniq VRAM — **ma'lumot topilmadi**)
- Tavsiya: **"at least 30 minutes of transcribed speech data"**
- Davomiyligi: "several hours to days"
- Natija raqamlari: **ma'lumot topilmadi**
- ⚠️ **"language support is limited to XTTS V2's existing language capabilities"** → **yangi til bermaydi**

### 4.5 D yo'li hukmi — KRITIK CHEKLOV

| Savol | Javob |
|---|---|
| LoRA **ovoz** uchun ishlaydimi? | **HA, tasdiqlangan.** F5 LoRA r32 to'liq FT'dan WER va UTMOS bo'yicha yaxshi. CosyVoice3 LoRA to'liq SFT muvaffaqiyatsiz bo'lgan joyda ishladi. |
| LoRA **yangi til** uchun ishlaydimi? | ⚠️ **Nashr etilgan dalil TOPILMADI.** Barcha F5/CosyVoice LoRA hisobotlari **bitta ovoz adaptatsiyasi**. XTTS LoRA repo'si ochiq aytadi: yangi til qo'shmaydi. Chatterbox-finetuning vocab extension qiladi lekin LoRA'ni faqat "≤10 hours" datasetlar uchun tavsiya qiladi (yangi til uchun >10h → to'liq FT). |
| Qisman yo'l bormi? | 📄 [UtterTune (arXiv 2508.09767)](https://arxiv.org/pdf/2508.09767) — "LoRA-Based **Target-Language Pronunciation Edit and Control** in Multilingual TTS". Bu aynan LoRA bilan **mavjud ko'p tilli modelning talaffuzini** yangi tilga to'g'rilash. ⚠️ **To'liq matn o'qilmadi, raqamlar ma'lumot topilmadi** — lekin yo'nalish to'g'ri. |
| VRAM | **CosyVoice3 LoRA peak 6.96 GB** — RTX 3050 8GB **chegarada ishlaydi**. F5 LoRA RTX 3090 Ti 24GB da tasdiqlangan (pastroq VRAM sinalmagan). |
| Narx | 2 soat data + 500 update → 🧮 bir necha soat 3090 da → **$1–$5**. 200 epoch CosyVoice3 → 🧮 **$5–$20**. |

> **Xulosa: LoRA yangi TIL uchun yagona yechim EMAS, lekin yangi til fine-tune'idan keyin OVOZ sifatini pardozlash uchun ideal.** Strategiya: avval to'liq (yoki katta) fine-tune bilan tilni o'rgatish (B yo'li), keyin LoRA bilan har bir maqsadli ovozga arzon adaptatsiya.

---

## 5. Cloud GPU iqtisodiyoti

### 5.1 Kuzatilgan narxlar

**Barcha narxlar 2026-10-01 kuni kuzatildi.**

#### RunPod — 💰 [runpod.io/pricing](https://www.runpod.io/pricing), 2026-10-01

| GPU | Community Cloud | Secure Cloud |
|---|---|---|
| **A100 80GB PCIe** | **$1.19/soat** | $1.59/soat |
| **A100 80GB SXM** | **$1.39/soat** | $1.59/soat |
| **H100 PCIe** | **$1.99/soat** | $2.89/soat |
| **H100 SXM** | $2.69/soat | $3.49/soat |
| H100 NVL | $2.59/soat | $3.19/soat |
| **RTX 4090** | **$0.34/soat** | $0.74/soat |
| **RTX 3090** | **$0.22/soat** | $0.50/soat |
| L40S | $0.79/soat | $1.09/soat |
| A40 | $0.35/soat | $0.49/soat |

#### Lambda — 💰 [lambda.ai/pricing](https://lambda.ai/pricing), 2026-10-01

| GPU | $/GPU/soat |
|---|---|
| B200 SXM6 (8x) | $6.69 |
| **H100 SXM (8x)** | **$3.99** |
| GH200 (1x) | $2.29 |
| **A100 SXM 80GB (8x)** | **$2.79** |
| A100 SXM 40GB (8x) | $1.99 |
| A100 PCIe 40GB | $1.99 |
| A10 (1x) | $1.29 |
| A6000 | $1.09 |
| **V100** | **$0.79** |
| Quadro RTX 6000 (1x) | $0.69 |

*"plus applicable sales tax/VAT/GST", self-serve on-demand, "first-come access".*

#### Vast.ai — 💰 [computeprices.com/providers/vast](https://computeprices.com/providers/vast), "Prices last updated: October 1, 2026"

| GPU | $/soat |
|---|---|
| A10 | $0.243 (1×24GB) |
| **A100 PCIE** | **$0.464 – $0.735** |
| **A100 SXM** | **$0.403 – $0.801** |

Vast.ai rasmiy 💰 [vast.ai/pricing](https://vast.ai/pricing) sahifasidan raqamlar olinmadi (dinamik yuklanadi) — lekin uchta narx modeli: **On-Demand** (sekundlik billing, kafolatlangan uptime), **Interruptible** (**50%+ arzonroq**, preemptible), **Reserved** (1–6 oy, **50% gacha chegirma**).

#### Agregator — 💰 [gpuperhour.com](https://gpuperhour.com/), "last updated October 1, 2026 at 07:59 UTC"

| GPU | Eng arzon | Provayder |
|---|---|---|
| H100 SXM5 | **$2.79/soat** | Lyceum |
| **A100 SXM4 80GB** | **$0.67/soat** | **Vast.ai** |
| **RTX 4090** | **$0.43/soat** | **Vast.ai** |
| **RTX 3090** | **$0.39/soat** | **Vast.ai** |
| L40S | $0.97/soat | Massed Compute |

💰 [gpuperhour.com/rent/rtx-4090](https://gpuperhour.com/rent/rtx-4090), 2026-10-01: Vast.ai **$0.43**, RunPod **$0.74**, LeaderGPU **$1.80–$2.50**. "pricing ranges from $0.43/GPU/hr to $2.50/GPU/hr across 8 offers from 3 providers." Uzluksiz oylik: **~$307**.

#### ⚠️ MANBALAR NIZOSI (4090 narxi)

| Manba | Vast.ai 4090 | Sana |
|---|---|---|
| gpuperhour.com | **$0.43/soat** | 2026-10-01 |
| synpixcloud blog | **$0.39/soat** ("typical", diapazon $0.29–$0.59) | 2026-04 |
| thundercompute blog | **$0.56/soat** (on-demand) | 2026-09 |
| boshqa agregator | **$0.72/soat** | — |

| Manba | RunPod 4090 | Izoh |
|---|---|---|
| RunPod rasmiy | **$0.34** (Community) / **$0.74** (Secure) | ikki tier |
| gpuperhour | **$0.74** | faqat Secure'ni ko'rsatgan |

**Rejalashtirish uchun:** 4090 = **$0.34–$0.74/soat**, 3090 = **$0.22–$0.39/soat**, A100 80GB = **$0.67–$1.59/soat**. Vast.ai marketplace — narx host'larga qarab o'zgaradi, deployment vaqtida tekshirish shart.

### 5.2 Hisoblangan narxlar

#### (i) Zero-shot modelni yangi tilga fine-tune (~10–20 soat ma'lumot)

**Eng yaqin real anchorlar:**

| Anchor | Ma'lumot | GPU | Vaqt | 🧮 Narx (eng arzon) | 🧮 Narx (eng qimmat) |
|---|---|---|---|---|---|
| **lpscr (yunon)** 💬 #57 | 40h+20h | 1× RTX 3090 | **~12 soat** | 12 × $0.22 = **$2.64** (RunPod Community) | 12 × $0.39 = **$4.68** (Vast) |
| **Polyak referens** 💬 #1168 | **~90 soat** | 1× A100 80GB | **~24 soat** | 24 × $0.67 = **$16.08** (Vast) | 24 × $1.59 = **$38.16** (RunPod Secure) |
| **healthyfat tavsiyasi** 💬 #1168 | — | 1× RTX 4090 | **2–5 kun = 48–120 soat** | 48 × $0.34 = **$16.32** | 120 × $0.74 = **$88.80** |
| **jpgallegoar (ispan)** 💬 #57 | 218h | 1× RTX 4090 | 6h51m/epoch | 1 epoch: 6.85 × $0.34 = **$2.33** | 6.85 × $0.74 = **$5.07** |
| **alien79 (italyan)** 💬 (ishonchlilik past) | — | **8× RTX 4090** | **73+ soat** = **584 GPU-soat** | 584 × $0.34 = **$199** | 584 × $0.74 = **$432** |

> **🧮 10–20 soat ma'lumot uchun realistik byudjet:**
>
> 10–20 soat **F5-TTS jamoa minimumidan (50h) past** — bitta urinish yetmaydi. Lekin GPU narxi jihatidan:
> - Bitta 4090 run, 100k qadam, 2–3 kun: **48–72 soat × $0.34–$0.74 = $16–$53**
> - Realistik: **3 urinish** (hyperparameter, dataset tozalash, vocab o'lchami) → **$50–$160**
> - Storage + egress + failed run'lar: **+$20–$40**
> - **JAMI: $70–$200** bitta ishlaydigan yangi-til fine-tune uchun.
>
> **Agar RTX 3050 8GB da qilinsa (batch kichik, grad_accum katta):** 💬 #143 da RTX 3080 10GB **300K qadam** ni bajargan. 🧮 3050 taxminan 2× sekinroq → **4–8 hafta uzluksiz**, **$0 GPU narxi** (+ elektr). Bu haqiqiy variant, faqat sabr kerak.

#### (ii) F5-TTS-Small sinfidagi modelni to'liq noldan o'qitish

🧮 (4-bo'limdagi hisob takrorlanadi)

| Stsenariy | A100-soat | Vast.ai ($0.67) | RunPod Community ($1.19) | Lambda ($2.79) |
|---|---|---|---|---|
| Optimistik (param+batch+update masshtab) | **207** | **$139** | **$246** | **$577** |
| Konservativ (batch+update) | **448** | **$300** | **$533** | **$1 250** |
| Eng konservativ (shipped config, to'liq batch) | **896** | **$600** | **$1 066** | **$2 500** |

Arifmetika: base = 8 GPU × 7 kun × 24 soat = **1 344 A100-soat**.
- Optimistik: 1 344 × (155/335.8) × 0.5 × (800/1200) = **207**
- Konservativ: 1 344 × 0.5 × (800/1200) = **448**
- Eng konservativ: 1 344 × (800/1200) = **896**

> **Lekin:** bu **945 soat ma'lumot** uchun. Yolg'iz dasturchi uchun **GPU $300 muammo emas — 945 soat o'zbek audio muammo.**

#### (iii) C1 yo'li (SC-GlowTTS / YourTTS sinfi) — eng arzon noldan

| Model | Ma'lumot | 🧮 GPU-soat | Vast 3090 ($0.39) | RunPod 3090 ($0.22) | Lambda V100 ($0.79) |
|---|---|---|---|---|---|
| **SC-GlowTTS** (150K qadam, batch 128) | **44 soat, 98 spikr** | **40–70** | **$16–$27** | **$9–$15** | **$32–$55** |
| **YourTTS** (~1.39M qadam, batch 64) | VCTK 44h + LibriTTS + 10h PT | **~257** | **~$100** | **~$57** | **~$203** |

> **SC-GlowTTS: 44 soat ma'lumot, ~$9–$55 GPU, SECS 0.8046 / MOS 3.78.** Bu **nashr etilgan eng arzon zero-shot retsepti.**

---

## 6. MUVAFFAQIYATSIZLIK HISOBOTLARI (eng qimmatli ma'lumot)

Bu bo'limni e'tibor bilan o'qing — bu pul va vaqt tejaydi.

| # | Kim / qayerda | Nima qilgan | Nima bo'lgan | Sabab (muallif so'zlari) |
|---|---|---|---|---|
| 1 | **costcuttingcz** 💬 [F5-TTS #453](https://github.com/SWivid/F5-TTS/discussions/453) | Chex tilini **noldan** o'qitish | **MUVAFFAQIYATSIZ** | **"I was not able do it from scratch"**; **"you will have to do finetune on existing models"**; **"2 hours is not inaf by my oppinion"**; fonetik xato: "not able to get rid of fonetically wrong reading of 'J' and also skipping 'ě'" |
| 2 | 💬 [F5-TTS #1011](https://github.com/SWivid/F5-TTS/discussions/1011) | **Arab tili, 61 soat yuqori sifatli, 400 000 update**, char-level | **MUVAFFAQIYATSIZ (overfit)** | O'qitish ma'lumotida **"pronunciation reaches around 85% accuracy at the character level"**, lekin ko'rilmagan matnda **"inaccurate and unintelligible"**. Klassik generalizatsiya muvaffaqiyatsizligi. Savol: char-level o'rniga phoneme-level kerakmi? |
| 3 | **bensonbs** 💬 [F5-TTS #57](https://github.com/SWivid/F5-TTS/discussions/57) | Xitoy, **~33 soat**, `lr=7.5e-5`, loss ~0.75 | **SIFAT YOMONLASHDI** | **"as the training steps increase, the pronunciation of words is becoming increasingly unclear"** — `model_12620` → `model_126200` da degradatsiya. Sabab: juda yuqori lr (base lr'ni ishlatgan, fine-tune uchun 1e-5 kerak edi — lpscr 1e-5 bilan muvaffaqiyat qozongan) |
| 4 | **instavar** 💬 [cosyvoice3-lora](https://github.com/instavar/cosyvoice3-lora-finetuning) | CosyVoice3 **to'liq SFT, 506M parametr**, 16 535 utterance, RTX 3090 Ti | **MUVAFFAQIYATSIZ** | **"Massive overfitting after epoch 1"**; **"At epochs 8-10, generating >10 second audio required 11-18 seed attempts"**; **"12+ evaluation directories tried different epochs... None produced production quality"**; 174 GB disk sarflandi; epoch 41 gacha gating'siz ketdi |
| 5 | 💬 [F5-TTS #769](https://github.com/SWivid/F5-TTS/discussions/769) | 12GB VRAM, fine-tune checkbox'ni belgilamagan | **NOLDAN O'QITIB QO'YDI** | **"output started as noise and is only slowly becoming speech"**. Yana: `max_samples > 8` → "horribly slow training speed". Yana: **"an error (not visible in the gradio interface) caused the fine-tuned model to remain loaded"** → **noto'g'ri baholash** (aslida fine-tune ishlagan edi!) |
| 6 | 💬 [F5-TTS #1132](https://github.com/SWivid/F5-TTS/discussions/1132) | **1 soat 15 daq** ingliz (825 sample), **Nvidia 4070 Laptop 8GB**, 30k qadam | **ARTEFAKT** | **"a wet sound, gargling like"**. Kuniga faqat **20k–25k qadam**. Yagona javob: **"Think This repo dies as no ones replying here anymore"** — thread javobsiz qoldi |
| 7 | **sanebg** 💬 [coqui #3689](https://github.com/coqui-ai/TTS/discussions/3689) | Bolgar tili, **48 soat studio audio**, **2× RTX 4090** | **NATIJA JOYLANMAGAN** | Thread javobsiz; **birorta ham muvaffaqiyat hisoboti yo'q**; keyinchalik F5-TTS ga o'tish tavsiya qilingan |
| 8 | Coqui maintainer'lar 💬 [coqui #3992](https://github.com/coqui-ai/TTS/issues/3992) | Yangi til fine-tune funksiyasi so'rovi | **"wontfix"** | Rasmiy yechim bo'lmaydi: "feel free to help" |
| 9 | 💬 [anhnh2002 repo](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages) | XTTS **HiFiGAN dekoderni fine-tune** | **YOMONLASHDI** | "HiFiGAN decoder finetuning reportedly worsened performance and isn't recommended" |
| 10 | 💬 [F5-TTS #143](https://github.com/SWivid/F5-TTS/discussions/143) | Bitta spikrda fine-tune | **ZERO-SHOT YO'QOLDI** | **"Trained models struggled to mimic different voices when fine-tuned on only one speaker's data"** |
| 11 | 💬 [Fish-Speech docs](https://speech.fish.audio/finetune/) | RL o'qitilgan modelni fine-tune | **DEGRADATSIYA** | "Fine-tuning a model after RL can shift the model distribution, which may lead to degraded performance" |
| 12 | 💬 [Spark-TTS discussions/1](https://huggingface.co/SparkAudio/Spark-TTS-0.5B/discussions/1) | Yangi til hujjati so'rovi | **JAVOB YO'Q** | Maintainer'lar javob bermagan; gruzin tili so'rovi ham javobsiz |

### Xulosa qilib olinadigan saboqlar

1. **Noldan o'qitish ishlamaydi** (#1, #5) — har doim pretrained checkpoint'dan fine-tune qiling. `model_1200000.pt` ni ko'chirib oling (💬 #57 da bir necha odam buni ta'kidlagan).
2. **Fine-tune lr = 1e-5, base lr = 7.5e-5 emas** (#3 vs lpscr muvaffaqiyati).
3. **Char-level 61 soatda yetarli emas** (#2) — generalizatsiya uchun phoneme-level yoki ko'proq matn xilma-xilligi kerak. 💬 [ablt.dev](https://www.ablt.dev/blog/turkish-tts/) ham buni tasdiqlaydi: **"A 10-hour dataset with high lexical variety will outperform a 30-hour monotone corpus."**
4. **To'liq SFT bitta spikrda = overfitting** (#4) — LoRA ishlatish.
5. **Bitta spikrda fine-tune zero-shot'ni o'ldiradi** (#10) — **sizning maqsadingiz 10 sekunddan klonlash bo'lsa, ko'p spikrli ma'lumot majburiy.**
6. **Baholashni ishonchli qiling** (#5) — gradio'dagi ko'rinmas xatolar noto'g'ri xulosaga olib keladi. `/src/f5_tts/eval/` vositalarini ishlatish tavsiya qilingan.
7. **EMA'ni o'chirish** erta fine-tune'da (💬 train README).
8. **Validation loss bo'yicha checkpoint tanlash + early stopping** (#4 tuzoq 2 va 3).

---

## 7. HALOL HUKM: har bir yo'l bo'yicha

### Yo'l A — Tayyor model
- **Vaqt:** 1 kun (o'rnatish)
- **Pul:** **$0**
- **Ma'lumot:** 0 (faqat 3–10 sek referens)
- **Sifat:** Qo'llanadigan tilda — **mukammal** (F5 WER 2.42%, XTTS CER 2.06). Qo'llanmaydigan tilda — **ishlamaydi** (talaffuz buzuq)
- **Kim uchun:** Agar tilingiz 16/17 ro'yxatda bo'lsa — **TO'XTANG, bu yetarli.** Aks holda — bu faqat baseline.

### Yo'l B1 — F5-TTS yangi tilga fine-tune ⭐ **TAVSIYA ETILADI**
- **Vaqt:** Ma'lumot tayyorlash **1–3 oy** (asosiy sarf). O'qitish **12 soat – 5 kun**. 2–3 urinish bilan jami **2–4 oy**.
- **Pul:** GPU **$16–$90** bitta run; realistik **$70–$200** (3 urinish + storage)
- **Ma'lumot:** **50 soat minimum, 100 soat yaxshi, 300+ optimal**, KO'P SPIKR
- **Sifat:** 12+ til tasdiqlangan. Tushunarli, tabiiy, lekin talaffuz xatolari qoladi (Tay: "reading long text or some words is still not accurate"). Venger muallifi normalizatsiya qatlami qo'shib tuzatgan.
- **Risk:** O'rtacha. Eng ko'p hujjatlashtirilgan yo'l, eng ko'p jamoa yordami.
- **VRAM:** RTX 3080 10GB da 300K qadam ishlagan → **RTX 3050 8GB chegarada, lekin mumkin** (batch kichik)
- **Litsenziya:** F5-TTS — ochiq (Emilia CC-BY-NC-4.0 cheklovi bilan tekshirilishi kerak)

### Yo'l B2 — XTTS-v2 yangi tilga fine-tune
- **Vaqt:** O'qitish — **nashr etilmagan**. Ma'lumot: 1–3 oy.
- **Pul:** 🧮 **$16–$52** (48–120 GPU-soat 4090da, mening ekstrapolyatsiyam)
- **Ma'lumot:** **~20 soat** (DVAE'siz GPU FT uchun) – 100h+ ("works well")
- **Sifat:** viXTTS/vnTTS mavjud va ishlaydi, lekin **birorta ham raqam nashr etmagan**
- **Risk:** **Yuqori.** Rasmiy protsedura yo'q ("wontfix"), javobsiz thread'lar, takrorlanadigan narx ma'lumoti yo'q
- **VRAM:** **16GB+ to'liq FT** → **RTX 3050 8GB YETMAYDI**
- **Litsenziya:** **CPML — nokommersiya.** Mahsulot qilish mumkin emas.

### Yo'l C1 — SC-GlowTTS / YourTTS sinfi (speaker encoder, noldan)
- **Vaqt:** O'qitish **40–260 GPU-soat**. Ma'lumot: 44 soat kerak.
- **Pul:** **$9–$100** (eng arzon nashr etilgan zero-shot)
- **Ma'lumot:** **44 soat, 98 spikr** (yoki 11 spikr bilan ham konvergensiya). Yangi tilni qo'shish uchun esa **5 daqiqa** yetishi mumkin (AACL 2022)
- **Sifat:** SC-GlowTTS **SECS 0.8046, MOS 3.78** / YourTTS **SECS 0.864, MOS 4.21**. Zamonaviy F5/XTTS'dan **sezilarli past** (2021 avlod), lekin **ishlaydigan zero-shot**
- **Risk:** O'rtacha. Kod eski (Coqui TTS arxivlangan), lekin ESPnet'da tirik retseptlar bor
- **VRAM:** V100 32GB batch 128 → **RTX 3050 8GB da batch 32 bilan mumkin**
- **Afzalligi:** **Speaker-encoder yo'li codec-LM yo'lidan ~10–20× arzon** (44 soat vs 945 soat; 150K qadam vs 800K qadam; 1 GPU vs 8 GPU)

### Yo'l C2 — F5TTS_Small noldan
- **Vaqt:** 207–896 A100-soat (bitta A100 da **9–37 kun**; 8× A100 da **1–5 kun**)
- **Pul:** **$139–$2 500** bitta run; realistik **$400–$7 500**
- **Ma'lumot:** **945 soat** (nashr etilgan konfiguratsiya)
- **Sifat:** **WER 4.17%, SIM-o 0.54** — ishlaydigan zero-shot
- **Risk:** **Juda yuqori.** 945 soat o'zbek audio yo'q.
- **Hukm:** **Amalda mumkin emas.** Pul emas, ma'lumot to'sqinlik qiladi.

### Yo'l D — LoRA / PEFT
- **Vaqt:** **bir necha soat** (F5: 500 update; CosyVoice3: 200 epoch)
- **Pul:** **$1–$20**
- **Ma'lumot:** **2 soat** (F5 LoRA tajribasi) – 10 soat (Chatterbox chegarasi)
- **Sifat:** **To'liq fine-tune'dan tengdek yoki yaxshiroq** (F5 LoRA r32: WER 4.5012 vs 4.7445; UTMOS 4.3475 vs 4.3046). CSP-FT: WER 8.49% vs 10.25%
- **Risk:** Past — lekin **yangi til uchun nashr etilgan dalil yo'q**
- **VRAM:** **6.96 GB peak** (CosyVoice3) → **RTX 3050 8GB ishlaydi**
- **Hukm:** **Yangi til uchun yagona yechim emas, lekin B yo'lidan keyin har bir ovozga adaptatsiya uchun ideal.**

### YAKUNIY REYTING (yolg'iz dasturchi uchun)

| O'rin | Yo'l | Nega |
|---|---|---|
| 🥇 **1** | **B1: F5-TTS yangi tilga fine-tune** | Eng ko'p tasdiqlangan (12+ til), eng ochiq raqamlar, eng arzon run ($2.64 lpscr), RTX 3050 chegarada ishlaydi, kichik model (335.8M) |
| 🥈 **2** | **D: LoRA (B1 ustiga)** | $1–$20, 6.96GB VRAM, to'liq FT'dan yaxshi sifat. Til uchun emas, OVOZ uchun |
| 🥉 **3** | **C1: SC-GlowTTS/YourTTS** | Agar sizda 50 soatdan kam ma'lumot bo'lsa — **yagona haqiqiy noldan yo'l**. 44 soat, $9–$100. Sifat pastroq, lekin ishlaydi |
| 4 | **A: Tayyor model** | $0 baseline. Agar til qo'llanilsa — birinchi o'rin. Aks holda faqat qiyoslash uchun |
| 5 | **B2: XTTS-v2** | Ishlaydi (viXTTS), lekin 16GB+ VRAM, nokommersiya litsenziya, raqamlar yopiq, "wontfix" |
| ❌ | **C2: F5TTS_Small noldan** | 945 soat ma'lumot — amalda mumkin emas |

---

## 8. AMALIY REJA (eng arzon haqiqiy yo'l)

### 0-qadam: Baseline o'rnatish ($0, 1 kun)
Tayyor F5-TTS, XTTS-v2, CosyVoice2 ni RTX 3050 da ishga tushiring. Maqsadli tilda sinab ko'ring. **Yozib qo'ying: qancha yomon?** Keyinchalik taqqoslash uchun kerak.

### 1-qadam: Ma'lumot (asosiy xarajat, 1–3 oy, $0–$X)
- **Maqsad: 50–100 soat, KO'P SPIKR** (minimum 20–30 spikr; YourTTS 1 151 spikr ko'rgan)
- ⚠️ **Bitta spikrda fine-tune qilmang — zero-shot yo'qoladi** (💬 #143)
- Mono WAV, **24kHz (Vocos)** yoki **44kHz (BigVGAN)** (💬 #143)
- **Leksik xilma-xillik > soat** (💬 ablt.dev: 10h xilma-xil > 30h monoton)
- Manba: Mozilla Common Voice (💬 ablt.dev tavsiyasi) + audiokitob + o'z yozuvlar

### 2-qadam: F5-TTS fine-tune ($16–$90 bitta run)
```
Base:        SWivid/F5-TTS  model_1200000.pt  (ko'chirib oling!)
Tokenizer:   char-level, vocab extension (yangi harflar qo'shish)
lr:          1e-5   (7.5e-5 EMAS — bensonbs muvaffaqiyatsizligi)
use_ema:     False  (erta fine-tune'da)
batch:       3050 8GB da ~1600, grad_accum katta
Qadam:       10-20k = dastlabki tushunarlilik; 100-150k = ishlab chiqarish
GPU:         Vast.ai RTX 4090 $0.43/h  yoki  RunPod 4090 Community $0.34/h
             (katta batch kerak bo'lsa: Vast A100 80GB $0.67/h)
Vaqt:        2-5 kun (4090) / ~24 soat (A100, 90h data)
Baholash:    /src/f5_tts/eval/  —  gradio'ga ishonmang
```
**Narx: 48–120 soat × $0.34–$0.74 = $16–$89 bitta urinish.**

### 3-qadam: LoRA bilan ovozlarni pardozlash ($1–$20)
Til o'rgangandan keyin har bir maqsadli ovozga LoRA (rank 16–32). **RTX 3050 8GB da mumkin** (6.96 GB peak). 💬 [1109-christine/F5-TTS-LoRA](https://github.com/1109-christine/F5-TTS-LoRA) retseptini ishlatish.

### 4-qadam: Agar 50 soat yo'q bo'lsa — C1 ga o'tish
50 soatdan kam bo'lsa, F5-TTS yangi-til fine-tune qilmang (💬 #453: "2 hours is not inaf"; #1011: 61h char-level overfit). O'rniga:
- SC-GlowTTS / YourTTS retsepti (ESPnet yoki Coqui arxiv kodi)
- **44 soat, 98 spikr → SECS 0.8046, MOS 3.78, $9–$55**
- Yoki 📄 AACL 2022 yondashuvi: ko'p tilli baza + speaker encoder + yangi tilga 5 daqiqa

### TO'LIQ BYUDJET

| Element | Narx |
|---|---|
| Baseline sinov | **$0** (RTX 3050) |
| Ma'lumot yig'ish/tozalash | **$0** (o'z vaqtingiz) – $X (yollangan ovoz aktyorlari) |
| F5-TTS fine-tune, 1-urinish | **$16–$89** |
| 2 qo'shimcha urinish | **$32–$178** |
| Muvaffaqiyatsiz run'lar, storage, egress | **$20–$40** |
| LoRA ovoz adaptatsiyalari (×5 ovoz) | **$5–$100** |
| **JAMI GPU BYUDJETI** | **$73–$407** |
| **Realistik maqsadli raqam** | **~$150–$250** |

---

## 9. Qo'shimcha: nizolar va aniqlanmagan nuqtalar

### Manbalar nizosi

1. **F5TTS_Small batch size:** 📄 Maqola — "half the base model batch size"; 💬 shipped `F5TTS_Small.yaml` — `38400/GPU × 8 = 307 200` (base bilan bir xil). **Ikkisi ham keltirildi.**
2. **`hotstone228/F5-TTS-Russian` ma'lumoti:** model card "100k hours" deydi — bu base F5-TTS Emilia raqami, rus ma'lumoti emas. Bir xil card'da Common Voice 17.0 + SOVA + Golos ko'rsatilgan (bu ~1–2k soat tartibida). **Card ichida ziddiyat.**
3. **`jpgallegoar/F5-Spanish` "1,200,000 steps":** base checkpoint nomi ham `model_1200000.pt`. Bu yangi qadamlar yoki base hisoblagichi — **aniqlanmagan.**
4. **4090/A100 narxlari:** 5.1-bo'limdagi nizo jadvali.
5. **XTTS til soni:** docs 16 til; HF card + bir qancha manba 17 til (hindi qo'shilgan).

### Topilmagan ma'lumotlar (ma'lumot topilmadi)

- XTTS-v2 yangi-til fine-tune uchun **birorta ham wall-clock vaqt yoki GPU-soat**
- viXTTS, vnTTS, F5-TTS-Hun, F5-TTS-THAI, Misha24-10/F5-TTS_RUSSIAN uchun **GPU turi va o'qitish vaqti**
- Barcha F5-TTS til fine-tune'lari uchun **WER/CER/SIM raqamlari** (birortasi ham baholash metrikasi nashr etmagan!)
- **IndexTTS2** fine-tune protsedurasi, VRAM, o'qitish soatlari
- **Fish-Speech** minimal dataset, VRAM, LoRA vs full FT
- **Spark-TTS** yangi til hujjati
- **Amphion** zero-shot multi-speaker + speaker-encoder retsepti
- **Matcha-TTS + speaker encoder** nashr etilgan zero-shot ishi
- **Zero-Voice** to'liq matni (HTTP 403)
- **BnTTS** raqamlari (PDF o'qilmadi)
- **YourTTS** wall-clock o'qitish vaqti
- **LoRA bilan yangi til qo'shish** bo'yicha nashr etilgan metrikalar (UtterTune bor, lekin matni o'qilmadi)
- XTTS maqolasining **har til bo'yicha soat taqsimoti** (faqat jami 27.3k soat / 16 til ma'lum)

### Byudjet cheklovi haqida ogohlantirish

Bu tadqiqot **13 WebSearch** chaqiruvi ishlatdi (ruxsat 18 ta). Qolgan ma'lumotlar **WebFetch** orqali to'g'ridan-to'g'ri olindi. Bir qancha arXiv PDF fayllari binary sifatida qaytdi va o'qilmadi (YourTTS va F5-TTS uchun ar5iv/HTML muqobili topildi; BnTTS, AACL 2022 va XTTS maqolalari uchun qisman yoki umuman topilmadi).

---

## 10. Barcha manbalar

### Maqolalar 📄
- F5-TTS: https://arxiv.org/abs/2410.06885 | https://arxiv.org/html/2410.06885v3
- YourTTS: https://arxiv.org/abs/2112.02418 | https://ar5iv.labs.arxiv.org/html/2112.02418
- SC-GlowTTS: https://arxiv.org/abs/2104.05557 | https://ar5iv.labs.arxiv.org/html/2104.05557
- XTTS (Interspeech 2024): https://www.isca-archive.org/interspeech_2024/casanova24_interspeech.html | https://arxiv.org/pdf/2406.04904
- CosyVoice 2: https://arxiv.org/html/2412.10117v1
- Low-Resource Multilingual and Zero-Shot Multispeaker TTS (AACL 2022): https://aclanthology.org/2022.aacl-main.56.pdf
- Characteristic-Specific Partial Fine-Tuning (CSP-FT): https://arxiv.org/html/2501.14273v1
- Speech LLMs in Low-Resource Scenarios: https://arxiv.org/pdf/2508.05149
- Exploring Pre-training Benefits on Phoneme Addition: https://arxiv.org/pdf/2606.19792
- UtterTune (LoRA pronunciation edit): https://arxiv.org/pdf/2508.09767
- kNN Retrieval Zero-Shot Multi-speaker TTS: https://arxiv.org/pdf/2408.10771
- BnTTS (NAACL 2025 Findings): https://aclanthology.org/2025.findings-naacl.279.pdf
- ESPnet2-TTS: https://ar5iv.labs.arxiv.org/html/2110.07840
- ESPnet-SPK: https://arxiv.org/html/2401.17230v2
- Zero-Voice (preprint, 403): https://www.techrxiv.org/users/882196/articles/1260796-zero-voice-a-low-resource-approach-for-zero-shot-text-to-speech
- Zero-Shot Voice Cloning with Minimal Data (tezis, PDF o'qilmadi): https://campus-fryslan.studenttheses.ub.rug.nl/708/1/MAs5965055QYZhu.pdf

### Jamoa hisobotlari 💬
**XTTS:**
- https://docs.coqui.ai/en/latest/models/xtts.html
- https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages
- https://github.com/coqui-ai/TTS/discussions/3689
- https://github.com/coqui-ai/TTS/issues/3992
- https://github.com/coqui-ai/TTS/discussions/3457
- https://github.com/gokhaneraslan/XTTS_V2-finetuning
- https://github.com/erew123/alltalk_tts
- https://huggingface.co/capleaf/viXTTS
- https://huggingface.co/anhnh2002/vnTTS
- https://www.emergentmind.com/topics/xtts-model

**F5-TTS:**
- https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/train/README.md
- https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/configs
- https://github.com/SWivid/F5-TTS/discussions/57
- https://github.com/SWivid/F5-TTS/discussions/143
- https://github.com/SWivid/F5-TTS/discussions/453
- https://github.com/SWivid/F5-TTS/discussions/614
- https://github.com/SWivid/F5-TTS/discussions/769
- https://github.com/SWivid/F5-TTS/discussions/1011
- https://github.com/SWivid/F5-TTS/discussions/1132
- https://github.com/SWivid/F5-TTS/discussions/1168
- https://huggingface.co/jpgallegoar/F5-Spanish
- https://github.com/jpgallegoar/Spanish-F5
- https://huggingface.co/hotstone228/F5-TTS-Russian
- https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN
- https://huggingface.co/VIZINTZOR/F5-TTS-THAI
- https://huggingface.co/sarpba/F5-TTS-Hun
- https://huggingface.co/aihpi/F5-TTS-German
- https://huggingface.co/models?search=F5-TTS

**LoRA / PEFT:**
- https://github.com/1109-christine/F5-TTS-LoRA
- https://github.com/instavar/f5-tts-lora-finetuning
- https://github.com/instavar/cosyvoice3-lora-finetuning
- https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning

**Boshqa modellar:**
- https://github.com/index-tts/index-tts
- https://speech.fish.audio/finetune/
- https://huggingface.co/blog/lengyue233/fish-speech-1
- https://huggingface.co/SparkAudio/Spark-TTS-0.5B/discussions/1
- https://huggingface.co/HKUSTAudio/Llasa-3B
- https://huggingface.co/blog/Steveeeeeeen/llasagna
- https://github.com/zhenye234/LLaSA_training
- https://github.com/gokhaneraslan/chatterbox-finetuning
- https://github.com/RVC-Boss/GPT-SoVITS/issues/2595
- https://github.com/RVC-Boss/GPT-SoVITS/issues/241
- https://huggingface.co/datasets/amphion/Emilia-Dataset
- https://www.ablt.dev/blog/turkish-tts/
- https://www.emergentmind.com/topics/cosyvoice2

### Narx manbalari 💰 (barchasi 2026-10-01 kuzatilgan)
- https://www.runpod.io/pricing
- https://lambda.ai/pricing
- https://vast.ai/pricing
- https://computeprices.com/providers/vast
- https://gpuperhour.com/
- https://gpuperhour.com/rent/rtx-4090
- https://www.synpixcloud.com/blog/vast-ai-vs-runpod-rtx-4090-pricing
- https://www.thundercompute.com/blog/vast-ai-vs-thunder-compute

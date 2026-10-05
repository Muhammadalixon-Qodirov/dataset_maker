# O'zbek tili uchun zero-shot ovoz klonlash: qaysi baza real, va o'zbekni qo'shish aniq nimani talab qiladi

**Tadqiqot sanasi:** 2026-10-01
**Qamrov:** faqat o'zbek/turkiy jihat. Arxitektura ichki tuzilishi, global model katalogi, arzon o'qitish yo'llari va VRAM — boshqa agentlar zimmasida.

---

## 0. Eng muhim xulosa (oldindan)

Tadqiqotning boshida kutilgan javob «o'zbek hech qayerda yo'q, demak o'zimiz qo'shishimiz kerak» edi. **Bu javob endi to'g'ri emas.**

**OmniVoice** (k2-fsa / Xiaomi, 2026) — 646 tilni qo'llab-quvvatlaydigan zero-shot ovoz klonlash modeli — **o'zbek tilini rasmiy ro'yxatida saqlaydi**: `uz` / ISO `uzb`, **115.28 soat** o'qitish ma'lumoti bilan. Bundan tashqari alohida `uzn` (Northern Uzbek) yozuvi ham bor — 15.23 soat.

Va eng muhimi — **sifat o'lchangan**. FLEURS-Multilingual-102 benchmarkida o'zbek tili uchun:

| | CER (%) |
|---|---|
| Haqiqiy inson yozuvi (ground truth) | **3.91** |
| OmniVoice sintezi | **3.37** |

Ya'ni OmniVoice sintez qilgan o'zbek nutqi ASR o'lchovi bo'yicha FLEURS'dagi haqiqiy inson yozuvlaridan **tushunarliroq** chiqdi.

Manba: [arXiv:2604.00688 Table 10 va Table 11](https://arxiv.org/abs/2604.00688), [docs/languages.md](https://raw.githubusercontent.com/k2-fsa/OmniVoice/main/docs/languages.md)

Demak amaliy vazifa «o'zbekni noldan modelga qo'shish» emas, balki **«mavjud bazani o'zbek uchun tekshirish, kalibrlash va kerak bo'lsa yengil fine-tune qilish»**. Bu bir necha oylik ishdan bir necha haftalik ishga tushadi.

Pastda barcha tekshirilgan modellar, nima tasdiqlangani va nima tasdiqlanmagani ko'rsatilgan.

---

## 1. Barcha zero-shot modellar × (O'zbek? Turk? Litsenziya?)

Har bir qatorda til ro'yxati **aslida tekshirildi** — model kartasi yoki rasmiy README'dan. «Multilingual» deb yozib, ro'yxat bermagan modellar aniq belgilangan.

| Model | O'zbek? | Turk? | Litsenziya | Tekshirilgan til ro'yxati / izoh |
|---|---|---|---|---|
| **OmniVoice** (k2-fsa, Xiaomi) | ✅ **HA** — `uz`/`uzb`, 115.28 s; `uzn` 15.23 s | ✅ `tr`/`tur`, 125.36 s | Kod: **Apache-2.0**; **vaznlar: CC-BY-NC** (Emilia cheklovi) | 646 til, 581k soat. [languages.md](https://raw.githubusercontent.com/k2-fsa/OmniVoice/main/docs/languages.md), [LICENSE](https://raw.githubusercontent.com/k2-fsa/OmniVoice/main/LICENSE), [HF card](https://huggingface.co/k2-fsa/OmniVoice) |
| **MMS-TTS** (Meta) | ⚠️ **Qisman** — faqat `uzb-script_cyrillic`. **Lotin varianti HuggingFace'da mavjud emas** | ✅ `tur` | **CC-BY-NC-4.0** | 1107 til, lekin **zero-shot klonlash YO'Q** — har til uchun alohida bir-ovozli VITS. [HF](https://huggingface.co/facebook/mms-tts-uzb-script_cyrillic) |
| **TurkicTTS** (IS2AI/ISSAI) | ✅ HA — lekin transliteratsiya orqali (pastga qarang) | ✅ | KazakhTTS2 korpusi: akademik + tijorat | 10 turkiy til, Tacotron2 + ParallelWaveGAN. **Klonlash emas**, bitta ovoz. [GitHub](https://github.com/IS2AI/TurkicTTS) |
| XTTS-v2 (Coqui) | ❌ **YO'Q** | ✅ `tr` | **Coqui Public Model License (CPML)** — tijoratga cheklov | 17 til: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi. [HF](https://huggingface.co/coqui/XTTS-v2) |
| Chatterbox Multilingual v3 (Resemble AI) | ❌ **YO'Q** | ✅ `tr` | **MIT** | 23 til: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh. [HF](https://huggingface.co/ResembleAI/chatterbox) |
| CosyVoice 2-0.5B | ❌ **YO'Q** | ❌ YO'Q | Apache-2.0 | 9 til: zh, en, ja, ko, de, es, fr, it, ru + 18 xitoy dialekti. [HF](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) |
| CosyVoice 3 | ❌ (ro'yxat e'lon qilinmagan) | ❌ | Maqolada ko'rsatilmagan | «9 languages and 18 Chinese dialects», 1 mln soat. **To'liq ro'yxat abstraktda yo'q — tasdiqlanmadi.** [arXiv:2505.17589](https://arxiv.org/abs/2505.17589) |
| Fish-Speech / OpenAudio S1-mini | ❌ **YO'Q** | ❌ YO'Q | **CC-BY-NC-SA-4.0** | 13 til: en, zh, ja, de, fr, es, ko, ar, ru, nl, it, pl, pt. [HF](https://huggingface.co/fishaudio/openaudio-s1-mini) |
| MaskGCT (Amphion) | ❌ **YO'Q** | ❌ YO'Q | **CC-BY-NC-4.0** | Emilia'da o'qitilgan, kartada faqat en + zh (har biri 50k soat). [HF](https://huggingface.co/amphion/MaskGCT) |
| F5-TTS | ❌ **YO'Q** | ❌ YO'Q | **CC-BY-NC-4.0** | Emilia (zh + en). Kartada til ro'yxati yo'q. 150+ community finetune bor. [HF](https://huggingface.co/SWivid/F5-TTS) |
| X-Voice (F5-TTS forki, 0.4B) | ❌ **YO'Q** | ❌ **YO'Q** | CC BY 4.0 (maqola) | 30 til ro'yxati tekshirildi: zh, id, ja, ko, th, vi, en, fr, de, it, pt, ru, es + 17 Yevropa tili. Turkiy til **umuman yo'q**. [arXiv:2605.05611](https://arxiv.org/abs/2605.05611) |
| IndexTTS2 / 2.5 | ❌ **YO'Q** | ❌ YO'Q | **bilibili Model Use License** | zh, en, ja, es, ar. [GitHub](https://github.com/index-tts/index-tts) |
| Zonos-v0.1 | ❌ o'qitilmagan, **lekin frontend'da `uz` bor** | ❌ | Apache-2.0 | en, ja, zh, fr, de da o'qitilgan. **espeak-ng fonemizatsiyasi** — bu muhim nuqta (4-bo'lim). [HF](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) |
| Orpheus (Canopy Labs) | ❌ **YO'Q** | ❌ YO'Q | Apache-2.0 (ingliz modellari) | «7 pairs» multilingual research preview — **README'da ro'yxat berilmagan**. [GitHub](https://github.com/canopyai/Orpheus-TTS) |
| Higgs Audio v2 | ❌ **YO'Q** | ❌ YO'Q | «other» | HF frontmatter til teglari: **en, zh, de, ko** — hammasi shu. 10 mln soat audio. [HF](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) |
| VibeVoice-1.5B (Microsoft) | ❌ **Ochiq-oydin YO'Q** | ❌ **Ochiq-oydin YO'Q** | MIT | Karta: «trained only on English and Chinese data; outputs in other languages are unsupported and may be unintelligible or offensive». [HF](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| Llasa-3B (HKUST) | ❌ **YO'Q** | ❌ YO'Q | **CC-BY-NC-4.0** | 250k soat zh-en. [HF](https://huggingface.co/HKUSTAudio/Llasa-3B) |
| Spark-TTS-0.5B | ❌ **YO'Q** | ❌ YO'Q | **CC-BY-NC-SA-4.0** | faqat zh + en, Qwen2.5 bazasida. [HF](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) |
| MegaTTS3 (ByteDance) | ❌ **YO'Q** | ❌ YO'Q | Apache-2.0 **lekin WaveVAE encoder vaznlari chiqarilmagan** | «For security issues, we do not upload the parameters of WaveVAE encoder» — o'z ovozingizni ishlatish uchun ularning navbatiga so'rov yuborish kerak. **Amalda ovoz klonlash erkin emas.** [HF](https://huggingface.co/ByteDance/MegaTTS3) |
| Parler-TTS large v1 | ❌ **YO'Q** | ❌ YO'Q | Apache-2.0 | Faqat ingliz. **Bu umuman zero-shot klonlash emas** — matnli tavsif bo'yicha ovoz (description-based). [HF](https://huggingface.co/parler-tts/parler-tts-large-v1) |

### 1-bo'lim xulosasi

- **18 ta asosiy zero-shot modeldan 16 tasida o'zbek YO'Q.** Bu ro'yxatlar faqat taxmin emas — har biri model kartasi/README'dan o'qildi.
- Turk tili ancha keng tarqalgan: XTTS-v2, Chatterbox Multilingual, MMS, TurkicTTS, OmniVoice — hammasida bor. Lekin turk tilining borligi o'zbekni **avtomatik** olib kelmaydi (3-bo'lim).
- **Faqat bitta model** — OmniVoice — o'zbekni zero-shot klonlash bilan birga, o'lchangan sifat bilan beradi.

---

## 2. OmniVoice — chuqur tekshiruv

Bu eng muhim topilma, shuning uchun batafsil.

### Nima ekani
- **To'liq nomi:** OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models
- **Maqola:** [arXiv:2604.00688](https://arxiv.org/abs/2604.00688), mualliflar: Han Zhu, Lingxuan Ye, Wei Kang, Zengwei Yao, Liyong Guo, Fangjun Kuang, Zhifeng Han, Weiji Zhuang, Long Lin, **Daniel Povey** (Kaldi/k2 asoschisi, Xiaomi)
- **Kod:** [github.com/k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
- **Vaznlar:** [huggingface.co/k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) — **1,431,614 yuklab olish, 1,456 like** (bu real, jiddiy ishlatilayotgan model, eksperiment emas)
- **Baza:** Qwen3-0.6B (0.6B parametr), Higgs-audio tokenizer (8 kodbukli akustik tokenlar)
- **Arxitektura:** diffusion language model uslubidagi bir bosqichli diskret non-autoregressive (NAR) — matndan to'g'ridan-to'g'ri akustik tokenlarga

### O'zbek tili uchun aniq raqamlar

| Ko'rsatkich | Qiymat | Manba |
|---|---|---|
| O'qitish ma'lumoti (`uz`/`uzb`) | **115.28 soat** | Table 11, languages.md |
| Qo'shimcha `uzn` (Northern Uzbek) | 15.23 soat | Table 11 |
| FLEURS uz — ground truth CER | 3.91% | Table 10 |
| FLEURS uz — OmniVoice CER | **3.37%** | Table 10 |
| 102 til bo'yicha o'rtacha SIM-o (ovoz o'xshashligi) | 0.788 | Table 4 |
| 102 til bo'yicha o'rtacha CER | 4.00% (ground truth 5.11%) | Table 4 |
| CER < 5% bo'lgan tillar soni | 82 / 102 (ground truth: 75) | Table 4 |

Qiyoslash uchun boshqa turkiy tillar (CER: ground truth → OmniVoice):

| Til | Soat | GT CER | OmniVoice CER |
|---|---|---|---|
| **O'zbek (uz)** | **115.28** | **3.91** | **3.37** ✅ yaxshiroq |
| Turk (tr) | 125.36 | 1.94 | 2.71 |
| Qozoq (kk) | 1537.29 | 2.01 | 3.11 |
| Qirg'iz (ky) | 46.63 | 2.33 | 1.80 ✅ yaxshiroq |
| Ozarbayjon (az) | 9.84 | 2.60 | 3.20 |
| Tatar (tt) | 30.03 | — | — |
| Turkman (tk) | 2.86 | — | — |

Diqqat qiling: qozoq tilida 1537 soat bo'lsa ham CER o'zbekdan yaxshiroq emas. Ya'ni bu modelda **~100 soat allaqachon yetarli darajada ishlaydi** — maqola buni ochiq aytadi: «OmniVoice maintains high intelligibility (CER < 5%) even for many languages with less than 10 hours of training data».

### Matn frontend — eng yaxshi yangilik

Maqoladan to'g'ridan-to'g'ri sitata:

> «For multilingual text processing, we employ the **subword tokenizer of pre-trained LLMs, eliminating cumbersome grapheme-to-phoneme conversion and language-specific text normalization**. This strategy also allows the LLM backbone to inherit and leverage its pre-trained knowledge efficiently.»

Buning ma'nosi o'zbek uchun:
- **Hech qanday G2P kerak emas.** espeak-ng kerak emas. Fonema lug'ati kerak emas.
- **O'zbek lotin matni to'g'ridan-to'g'ri kiradi** — Qwen3 tokenizeri orqali.
- **Tokenizerni kengaytirish kerak emas** → pretrained embeddingni buzish riski yo'q. Bu XTTS-v2 va Chatterbox'dagi eng katta texnik to'siqni butunlay chetlab o'tadi.
- Faqat xitoy (pinyin) va ingliz (CMU) uchun ixtiyoriy fonetik override mavjud — o'zbek uchun bu kerak emas.

### Ma'lumot manbalari (581k soat, 50 ta dataset)

To'liq ro'yxat (Appendix A): Emilia, Emilia-YODAS, LibriTTS, **Common Voice**, VoxBox, **Meta Omnilingual ASR Corpus**, **FLEURS**, **GigaSpeech 2**, YODAS-Granary, EuroSpeech, IndicVoices-R, IndicTTS, Rasa, PhoAudiobook, viVoice, CML-TTS, Wenetspeech-yue/chuan, KeSpeech, AESRC, KsponSpeech, ReazonSpeech, AISHELL6-whisper, ParaSpeechCaps, NVSpeech, NonVerbalSpeech-38K, SeniorTalk, ChildMandarin, **KSC2 (qozoq)**, ClArTTS, ArVoice, **KazakhTTS2**, RuLS, IMUT-MC, HUI-Audio-Corpus-German, MGB2/3/5, SADA, MASC, Mixat, arab datasetlari, NICT-Tib1, TIBMD@MUC, XBMU-AMDO3, Tibetan-Greetings.

O'zbekning 115 soati, ehtimol, Common Voice uz + Meta Omnilingual ASR Corpus + FLEURS + GigaSpeech 2 dan yig'ilgan. **USC va UzbekVoice ro'yxatda YO'Q** — ya'ni mahalliy eng yaxshi o'zbek korpuslari hali ishlatilmagan. Bu bizning qo'limizda qolgan asosiy ustunlik.

Shovqin bilan ishlash: «Given the heterogeneous quality of these sources (many of which were not originally designed for TTS and contain noisy audio or transcriptions), we employed a **speech restoration model** to enhance degraded speech and applied rule-based filtering to exclude invalid transcriptions.» Ya'ni ular ham ASR datasetlarini TTS uchun restavratsiya qilish yo'lidan borgan.

Kam resursli tillarni balanslash: `α = 0.8` smoothing bilan resampling — kam resursli tillar repetition factor bilan upsample qilinadi.

### Litsenziya — eng katta ogohlantirish

| Nima | Litsenziya |
|---|---|
| Kod (github.com/k2-fsa/OmniVoice) | **Apache-2.0** (Copyright: Xiaomi Corp., 2026) |
| Pretrained vaznlar (HF k2-fsa/OmniVoice) | **CC-BY-NC** — «due to constraints from its training data (e.g., Emilia)» |

**Bu tijorat mahsulot uchun jiddiy to'siq.** Vaznlar non-commercial. Qo'shimcha: README'da «Users are strictly prohibited from using this model for unauthorized voice cloning, voice impersonation, fraud, scams».

Variantlar:
- Tadqiqot / ichki demo / portfolio uchun — muammo yo'q.
- Tijorat uchun — yoki Resemble AI Chatterbox (MIT) yo'liga o'tish, yoki OmniVoice kodi (Apache-2.0) bilan **o'z vaznlarini noldan o'qitish** (bu juda qimmat), yoki k2-fsa bilan bog'lanish.
- `k2-fsa/OmniVoice-Emilia` alohida checkpoint ham bor (faqat 43 yuklab olish) — bu bilingual variant, litsenziya muammosini hal qilmaydi.

### Praktik tomoni
- `pip install omnivoice`
- CLI: `omnivoice-demo`, `omnivoice-infer`, `omnivoice-infer-batch`
- Til `language_id` parametri orqali beriladi (batch inference JSONL'da)
- RTF 0.025 gacha (real vaqtdan 40× tez), FlashInfer bilan 2–2.9× qo'shimcha
- NVIDIA GPU / Apple Silicon / Intel Arc XPU qo'llanadi
- Fine-tune skriptlari mavjud: `examples/run_finetune.sh`, `examples/run_finetune_lora.sh` — **LoRA fine-tune rasman qo'llanadi**

---

## 3. Turkiy tildan transfer tahlili — turk tilining borligi o'zbekka yordam beradimi?

### Yagona nashr etilgan to'g'ridan-to'g'ri dalil: TurkicTTS (INTERSPEECH 2023)

ISSAI (Nazarbayev University) 10 turkiy til uchun TTS qurdi: ozarbayjon, boshqird, **qozoq**, qirg'iz, saxa, tatar, turk, turkman, uyg'ur, **o'zbek**.

Metod: qozoq tilida o'qitilgan Tacotron2 modeli. Boshqa turkiy tillarning matni IPA'ga, keyin qozoq alifbosi harflariga aylantiriladi (transliteratsiya), va qozoq modeliga beriladi.

**Natijalar (o'zbek tili uchun), 22 baholovchi:**

| Ko'rsatkich | O'zbek | Umumiy diapazon (10 til) |
|---|---|---|
| Quality (Q, 1–5) | **2.85** | 2.37 – 4.18 |
| Comprehensibility (C, %) | **80%** | 45 – 97% |
| **Intelligibility (I, %)** | **45%** | 15 – 80% |

Manba: [IS2AI/TurkicTTS](https://github.com/IS2AI/TurkicTTS), [arXiv:2305.15749](https://arxiv.org/abs/2305.15749)

**Bu natija to'g'ridan-to'g'ri xulosa beradi:** turkiy tildan transliteratsiya orqali transfer o'zbek uchun **ishlamaydi**. Intelligibility 45% — ya'ni gaplarning yarmidan kamrog'i to'g'ri tushunildi. Quality 2.85/5 — mahsulot darajasidan ancha past. O'zbek 10 tilning o'rtasidan pastda turadi.

Sabab aniq: o'zbek fonologiyasi qozoqdan jiddiy farq qiladi (unlilar tizimi, singarmonizm darajasi, arab-fors qatlami), va IPA→qozoq-alifbo xaritasi bu farqlarni yo'qotadi.

### Turk tilining modelda borligi o'zbekka yordam beradimi?

**To'g'ridan-to'g'ri tajriba dalili topilmadi** — turk tilida o'qitilgan zero-shot modelni o'zbek matniga sinab ko'rgan nashr etilgan ish yo'q. Lekin bilvosita kuchli dalil bor:

1. **OmniVoice ma'lumotlari turk tilining yordam bermasligini ko'rsatadi.** OmniVoice'da turk ham (125 s), o'zbek ham (115 s) alohida mavjud. Agar turk tilidan transfer kuchli bo'lsa, 1537 soatli qozoq tili eng yaxshi CER berishi kerak edi — lekin qozoq CER 3.11, qirg'iz esa faqat 46 soat bilan 1.80. Ya'ni bu modelda **har til o'z ma'lumotidan o'rganadi**, oila a'zoligidan emas.

2. **Orfografik to'siq.** Turk alifbosi: `ç ş ğ ı ö ü`. O'zbek (1995) lotin alifbosi: `oʻ gʻ sh ch ng` + apostrof. Bir xil tovushlar **butunlay boshqacha yozilgan**. Turk tilida o'qitilgan tokenizer `sh` ni «s + h» deb, `oʻ` ni «o + apostrof» deb o'qiydi. Ya'ni turk tilining borligi hatto orfografik darajada ham o'tmaydi.

   **Qiziq nuqta:** 2026 alifbo islohoti aynan bu farqni yo'q qiladi (`sh→ş`, `ch→ç`, `gʻ→ğ`, `oʻ→ö`) — ya'ni islohotdan keyin o'zbek yozuvi turk yozuviga ancha yaqinlashadi va turk tilidan transfer **kelajakda** yaxshilanishi mumkin. Bu 9-bo'limda.

3. **Chatterbox community dalili (ijobiy tomon).** Chatterbox Multilingual'da 23 til bor, va jamoa **ro'yxatda bo'lmagan** tillarni muvaffaqiyatli qo'shgan: fors ([mazrba/Chatterbox-TTS-Persian-gguf](https://huggingface.co/mazrba/Chatterbox-TTS-Persian-gguf)), indonez ([grandhigh/Chatterbox-TTS-Indonesian](https://huggingface.co/grandhigh/Chatterbox-TTS-Indonesian)), norveg, daniya, fin, arab. Ya'ni yangi til qo'shish Chatterbox'da **takrorlangan, amalda isbotlangan jarayon** — faqat turkiy qarindoshlik emas, umumiy ko'p tilli baza yordam beradi.

### Xulosa (3-bo'lim)

- Turkiy tildan **transliteratsiya** orqali transfer — o'lchangan va **yetarli emas** (TurkicTTS: I=45%).
- Turk tilining model o'quv to'plamida bo'lishi o'zbekka **avtomatik yordam bermaydi**, chunki orfografiya mos kelmaydi va zamonaviy modellar til oilasi emas, o'z ma'lumotidan o'rganadi.
- Aslida yordam beradigan narsa — **ko'p tilli kenglik** (600 til), chunki model «har qanday lotin yozuvini o'qish» qobiliyatini umumlashtirgan.

---

## 4. «O'zbekni qo'shish» aniq nimani talab qiladi — 3 ta eng istiqbolli baza

### 4.1 OmniVoice (1-o'rin — tavsiya etiladi)

| Savol | Javob |
|---|---|
| Tokenizer / matn frontend | **Pretrained LLM (Qwen3-0.6B) subword tokenizeri.** G2P yo'q, espeak yo'q, til-spetsifik normalizatsiya yo'q |
| O'zbek lotin to'g'ridan-to'g'ri kiradimi? | **HA** |
| Tokenizerni kengaytirish kerakmi? | **YO'Q** → pretrained embeddingni buzish riski yo'q |
| O'zbek allaqachon bormi? | **HA, 115.28 soat** |
| Yangi til uchun qancha soat kerak? | Maqola kam resursli tillarda <10 soat bilan ham CER<5% ni ko'rsatadi; o'zbek allaqachon 115 soatda |
| Fine-tune qo'llanadimi? | HA — `examples/run_finetune.sh`, `examples/run_finetune_lora.sh` |
| **Qolgan ish** | (a) o'zbek sifatini o'z audiosi bilan sinash; (b) sonlar/qisqartmalar/chet so'zlar uchun matn normalizatsiyasi; (c) ixtiyoriy LoRA fine-tune — USC + Common Voice'ning toza qismi bilan sifatni ko'tarish; (d) alifbo normalizatsiyasi (`oʻ/o'/ó/ö` → bir xil shaklga) |
| **Qolgan risk** | **Vaznlar CC-BY-NC** — tijorat yo'q. Shuningdek `uz` ma'lumoti qaysi manbadan kelgani va u Lotin yoki Kirill bo'lgani maqolada ko'rsatilmagan — **bu amalda birinchi sinash kerak bo'lgan narsa** |

### 4.2 Chatterbox Multilingual v3 (2-o'rin — tijorat uchun eng yaxshi)

| Savol | Javob |
|---|---|
| Litsenziya | **MIT** — to'liq erkin, tijorat mumkin. Bu uning asosiy ustunligi |
| O'zbek bormi? | **YO'Q** (23 tilda yo'q) |
| Tokenizer | **Standard mode: grapheme (harf) tokenizeri, ~2,454 token, 23 tilni qamraydi.** Turbo mode: BPE ~50k ingliz vocab + ko'p tilli grapheme kengaytmasi |
| O'zbek lotin to'g'ridan-to'g'ri kiradimi? | **Deyarli** — grapheme tokenizer lotin harflarini tanidi. Mavjud o'zbek finetune kartasi aniq aytadi: «2454-token grapheme tokenizer **covers Uzbek's Latin script without modification from the base model**» |
| Tokenizerni kengaytirish kerakmi? | Standard mode'da — yo'q. Lekin **vocab size mos kelmasligi real muammo**: slovak finetune tajribasida «the fine-tuned model had different tokenizer vocab than the base model, causing shape mismatches when loading weights» |
| Yangi til uchun qancha soat? | Rasmiy ko'rsatma yo'q. Community toolkit: LoRA — ≤10 soat; full fine-tune — >10 soat «studio-grade». Arab misoli: 120 soat toza misrcha, 43,711 segment |
| **Allaqachon mavjud o'zbek finetune** | [Abduqayum/uzbek-tts-natural-speech-chatterbox](https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox) — LoRA, **~30 soat bir spikerli o'zbek audiokitob**, MIT. **Zero-shot klonlash saqlangan** |
| **Qolgan risk** | Sifat metrikasi nashr etilmagan; 30 soat bir ovoz — ko'p ovozli zero-shot buzilishi mumkin (6-bo'lim) |

### 4.3 XTTS-v2 (3-o'rin — eng ko'p yo'riqnoma, lekin eng qimmat yo'l)

| Savol | Javob |
|---|---|
| O'zbek bormi? | **YO'Q** (17 tilda yo'q). Turk bor |
| Tokenizer | **BPE** (byte-pair encoding) |
| Tokenizerni kengaytirish kerakmi? | **HA, albatta.** Jamoa yo'riqnomasida yangi til uchun vocab kengaytiriladi (vyetnam misolida +2000 token) |
| Yangi til uchun qancha soat? | **~48 soat tavsiya qilinadi; ba'zilar 100+ soat ishlatgan.** [Discussion #3689](https://github.com/coqui-ai/TTS/discussions/3689), [XTTSv2-Finetuning-for-New-Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages) |
| Litsenziya | **Coqui Public Model License (CPML)** — tijoratga cheklov, va Coqui kompaniyasi yopilgan |
| **Qolgan risk** | Eng ko'p ish, eng eski arxitektura, noqulay litsenziya. **Tavsiya etilmaydi** chunki OmniVoice'da o'zbek allaqachon bor |

### Alohida nuqta: Zonos va espeak-ng yo'li

Zonos-v0.1 **espeak-ng** bilan fonemizatsiya qiladi, va espeak-ng'da `uz` **mavjud**. Nazariy jihatdan bu «o'zbekni bepul qo'shish» yo'li ko'rinadi. **Amalda ishlamaydi:** Zonos faqat 5 tilda (en, ja, zh, fr, de) o'qitilgan, ya'ni model o'zbek fonemalar ketma-ketligini hech qachon ko'rmagan. espeak `uz` to'g'ri IPA bergan taqdirda ham model akustik tomondan noto'g'ri realizatsiya qiladi. Shuningdek espeak-ng `uz` sifati o'zi juda past (5-bo'lim).

---

## 5. espeak-ng `uz` = «status testing» — bu amalda nima degani

**Tasdiqlangan.** espeak-ng'ning o'zbek ovoz faylining **to'liq mazmuni** — uch qator:

```
name Uzbek
language uz

status testing
```

Manba: [espeak-ng-data/lang/trk/uz](https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/espeak-ng-data/lang/trk/uz)

espeak-ng'ning rasmiy ta'rifi bo'yicha ikki daraja bor ([docs/voices.md](https://github.com/espeak-ng/espeak-ng/blob/master/docs/voices.md)):
- **testing** — «Provides the basic level of support, with basic pronunciation rules. **Requires feedback from a native speaker** to help improve the voice.»
- **mature** — «The language is well supported. It has been improved by, or with support from, a native speaker.»

Ya'ni `uz` hali ona tili so'zlovchisi tomonidan tekshirilmagan.

### Amaliy holat — qoida faylini o'zi tekshirdim

[dictsource/uz_rules](https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/dictsource/uz_rules) fayli:
- **Jami ~90–100 qator.** (Qiyoslash uchun: yetuk tilning qoida fayli minglab qatordan iborat bo'ladi.)
- Tarkibi: `.replace` bo'limida **~30 ta kirill→lotin harf xaritasi**, keyin har harf uchun `.group` bo'limlari.
- Header: faqat `// Spelling to phoneme rules for Uzbek language`. **Muallif yo'q, ogohlantirish yo'q, cheklov ro'yxati yo'q.**
- espeak-ng docs/languages.md'da `uz` «[1,2]» bilan belgilangan — ya'ni lotin **va** kirill alifbosi.

### Ma'lum o'zbek xatolari (xulosa)

Nashr etilgan xatolar ro'yxati **topilmadi** — hech kim rasman hisobot qilmagan (bu «testing» statusining o'zi ko'rsatkichi). Lekin faylning strukturasidan kelib chiqib, kutilishi mumkin bo'lgan xatolar:
- 90 qator qoida o'zbekning allofonik o'zgarishlarini (unli garmoniyasi, so'z oxiridagi jarangsizlanish, `ng` birikmasi) qamrab olishi mumkin emas.
- `oʻ` va `gʻ` — apostrof variantlari (`'`, `ʻ`, `’`, `´`) ko'p xil yozilishi mumkin; 90 qatorli faylda bularning barchasi hal qilingani dargumon.
- Arab-fors va rus qatlamidagi so'zlar (masalan `maktab`, `kompyuter`) uchun alohida qoida yo'q.

### Xulosa (5-bo'lim)

**espeak-ng'ni o'zbek TTS frontendi sifatida ishlatish tavsiya etilmaydi.** Shuning uchun **espeak-ga tayanmaydigan** modellarni tanlash kerak — bu aynan OmniVoice (LLM subword tokenizeri) va Chatterbox (grapheme tokenizeri) ustunligi. Zonos esa espeak-ga tayangani uchun o'zbek yo'li sifatida tushib qoladi.

---

## 6. ENG KRITIK SAVOL: bir spikerli korpus (FeruzaSpeech, 59.61 s, bitta ayol) modelga YANGI TIL o'rgatish uchun yaroqlimi?

Qisqa javob: **Yaroqli, lekin yolg'iz o'zi EMAS.** Bu klassik, nashr etilgan muammo.

### Muammo rasman nima deb ataladi

Adabiyotda ikki xil nom bilan:
1. **Catastrophic forgetting** — bir spikerda fine-tune qilishda modelning ko'p ovozli xilma-xilligini yo'qotishi
2. **Speaker similarity / linguistic accuracy trade-off** — bir spikerli dataset bilan o'qitish «introduces a trade-off between speaker similarity and linguistic accuracy, where improving one aspect may degrade the other»

Manbalar: [Parameter-Efficient Fine-Tuning for Low-Resource TTS, INTERSPEECH 2025](https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf), [Bayesian PEFT for Overcoming Catastrophic Forgetting, arXiv:2402.12220](https://arxiv.org/pdf/2402.12220), [Residual Adapters for Few-Shot TTS Speaker Adaptation](https://lacuna.tiptreesystems.com/work/residual-adapters-for-few-shot-text-to-speech-speaker-adaptation/wrk_3885e6ec7a8dc1a03d280b298698b01e)

Muhim nuance: zero-shot TTS kontekstida sinash usuli — **OOD (out-of-domain) spikerlarda** ovoz o'xshashligini o'lchash. Agar maqsad spikerda SIM o'sib, OOD spikerlarda tushsa — bu pretrained bilimning yo'qolishi.

### Nima yordam beradi (nashr etilgan yechimlar)

| Yechim | Mohiyati | Dalil |
|---|---|---|
| **LoRA / adapterlar** | Baza modelni muzlatib, faqat kichik adapter o'qitiladi | «Prevents the model from forgetting its base knowledge (catastrophic forgetting)» — [chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning) |
| **Residual adapters** | Parametrlarning **~0.1%** — baza muzlagani uchun «the model retains the high-quality prosody of the original multi-speaker model» | Residual Adapters maqolasi |
| **Rehearsal / aralashtirish** | Yangi datasetga model allaqachon ko'rgan spikerlardan ma'lum nisbatda qo'shish | PEFT literaturasi |
| **Full fine-tune qilmaslik** | «ruining the base model's voice quality if the dataset is too small» | chatterbox-finetuning README |

### Eng qimmatli amaliy dalil — bu aynan o'zbek tilida qilingan

[Abduqayum/uzbek-tts-natural-speech-chatterbox](https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox):
- Chatterbox'ning **LoRA** finetune'i (keyin baza modelga merge qilingan, standalone T3 vaznlari)
- **~30 soat bir spikerli o'zbek audiokitob** o'qishi — ya'ni FeruzaSpeech'ga juda o'xshash sharoit
- **Natija: «zero-shot voice cloning remains functional. Users can pass any reference wav and it will speak your text in that voice»**
- Tokenizer o'zgartirilmagan
- MIT litsenziya

Va slovak tilidagi mustaqil tajriba ([7 Things That Mattered](https://dev.to/pekisko/fine-tuning-chatterbox-on-a-low-resource-language-7-things-that-mattered-13e1)) — Chatterbox'ning 23 tilida bo'lmagan til:
- **Ovoz klonlash fine-tune'dan keyin ham ishladi**
- **Ko'p tilli qobiliyatning catastrophic forgetting'i qayd etilmagan**
- Lekin real muammolar bo'ldi: tokenizer vocab mos kelmasligi (weight shape mismatch), reference audio sifati («whatever noise is in your reference clip will be baked into every generation»), birinchi so'z beqarorligi, generatsiya o'rtada to'xtab qolishi, qisqartma/birlik/harf nomlarini o'qiy olmasligi

### Xulosa (6-bo'lim)

**FeruzaSpeech (59.61 s, 1 ayol, CC BY 4.0) — o'zbekning eng toza audiosi, lekin uning roli «asos» emas, «sifat kalibratori».**

To'g'ri strategiya:
1. **Tilni** ko'p spikerli datadan o'rgatish (USC 958 spiker + Common Voice 2,299 spiker)
2. **Talaffuz va prosodiya sifatini** FeruzaSpeech bilan tekislash
3. **LoRA / adapter** ishlatish, full fine-tune emas
4. **Rehearsal:** o'zbek datasetiga baza modelning boshqa tillaridan ma'lum nisbatda qo'shish
5. **Sinash metodikasi:** fine-tune'dan keyin SIM-o ni **o'zbek bo'lmagan OOD spikerlarda** o'lchash — agar tushsa, forgetting boshlangan

Eng muhimi: **OmniVoice yo'lida bu savol umuman tug'ilmaydi**, chunki o'zbek allaqachon 115 soat ko'p spikerli datada o'qitilgan. FeruzaSpeech faqat ixtiyoriy yaxshilash uchun.

---

## 7. Mavjud o'zbek ko'p spikerli ma'lumotlar + TTS yaroqliligi bahosi

Zero-shot **ko'p** spiker talab qiladi, bitta emas. Mavjud hammasi:

| Dataset | Soat | Spiker | Sample rate | Yozuv | Litsenziya | TTS yaroqliligi | Kerakli filtrlash |
|---|---|---|---|---|---|---|---|
| **USC** (Uzbek Speech Corpus) | **104.9** (train 96.4 / valid 4.0 / test 4.5) | **958** (train 879) | Maqolada **ko'rsatilmagan**; WAV formatda | **Lotin, 29 harf + apostrof** (tasdiqlangan) | **CC BY 4.0** (maqola) / HF kartada **MIT** — ⚠️ **KELISHMOVCHILIK** | **O'rtacha-past.** Telegram bot orqali crowdsourced. Maqola ochiq aytadi: **«To make our dataset close to the real-world scenarios, we kept utterances containing background noises»** | Shovqin filtri majburiy (WV-MOS / NISQA), VAD, denoising. **Eng toza qism: 20 audiokitob o'quvchisi × 30 daqiqa ≈ 10 soat** — bu TTS uchun eng qimmatli bo'lak |
| **Common Voice uz** (cv-corpus-25.0-2026-03-09) | **265.75 jami / 101.01 validated** | **2,299** | **MP3**, o'zgaruvchan. Mozilla optimal deb 48 kHz ni ko'rsatadi | Lotin | CC0 | **Past (filtrlashsiz).** 230,040 klip: 87,442 validated (38.0%), 14,158 invalidated (6.2%), 128,440 «other» (55.8%). O'rtacha klip 4.159 s | Majburiy. Nashr etilgan dalil: WV-MOS filtri bilan **filtrlanmagan Common Voice'dan +1.26 MOS**, hatto studiyadagi LibriTTS'dan **+0.35 MOS** yaxshi natija ([arXiv:2210.06370](https://arxiv.org/abs/2210.06370)). Ma'lum muammolar: «mouse clicks, low-frequency noise, background speakers and music» |
| **UzbekVoice** (corpus.uzbekvoice.ai) | **>200** (1-bosqich, 2021-10..12) + **>1000** (2-bosqich, 2022-10..11); 3-bosqich rejada | **E'lon qilinmagan** | **E'lon qilinmagan** | Lotin | «open source» da'vo qilingan, **aniq litsenziya ko'rsatilmagan** | **Noma'lum** — texnik spetsifikatsiya e'lon qilinmagan. Asoschisi: Hasanov Bakhtiyor; hamkorlar: IT Park, UNDP | Baholash mumkin emas — avval ma'lumotni olish kerak |
| **uzbekvoice-filtered** (DavronSherbaev) | 501,330 train + 2,048 valid sample (soat ko'rsatilmagan; klip 0.47–12.6 s) | Metadata'da jins + 20 viloyat aksenti + tug'ilgan yil bor, **soni ko'rsatilmagan** | **16,000 Hz** ⚠️ | **Kirill** (namunalar kirillda) | **Apache-2.0** ✅ | **Past.** 16 kHz zamonaviy TTS uchun past (24 kHz minimum). Kirill yozuv qo'shimcha translit qadamini talab qiladi | Filtrlash allaqachon qilingan: VAD + noise detection, reading-speed outlier (~5–10%), avtomatik STT validatsiya. Lekin **16 kHz'ni tiklash mumkin emas** |
| **FLEURS uz_uz** | **~12** (tilga) | Juda kam (split boshiga bir nechta) | — | Lotin | **CC-BY-4.0** | **Faqat benchmark.** O'qitish uchun juda kichik | — |
| **FeruzaSpeech** | 59.61 | **1** (bitta ayol) | — | — | **CC BY 4.0** | **Audio sifati yuqori, lekin ko'p spikerli zero-shot uchun yaroqsiz** (6-bo'lim) | — |

### Jami hisob

| | Soat | Spiker |
|---|---|---|
| Ishonchli, ochiq, ko'p spikerli, lotin yozuvda | USC 104.9 + Common Voice validated 101.01 + FLEURS ~12 = **~218 soat** | 958 + 2,299 ≈ **~3,250** |
| Agressiv TTS filtrlashdan keyin kutilgan (taxminan 30–50%) | **~70–110 soat** | ~1,500–2,500 |
| UzbekVoice qo'shilsa (agar litsenziya va spetsifikatsiya ochiq bo'lsa) | +1,200 soat gacha | ? |

**Taqqoslash:** OmniVoice'da o'zbek allaqachon 115.28 soat. Ya'ni bizning qo'limizdagi filtrlangan ma'lumot **taxminan OmniVoice ishlatgan hajmga teng** — bu fine-tune uchun yetarli, lekin «sakrash» bermaydi. **Sakrash UzbekVoice'ning ~1200 soatini ochish orqali keladi.**

### Bu datasetlardan o'zbek TTS qurgan birortasi bormi?

- **USC muallifi o'zi ruxsat beradi:** «The USC is **primarily designed for the ASR task**, however, it can also be used to aid other speech-related tasks, such as **speech synthesis** and speech translation» ([arXiv:2107.14419](https://arxiv.org/abs/2107.14419))
- GitHub'da USC yoki Common Voice bilan o'zbek TTS qurgan loyihalar bor, lekin barchasi juda kichik: [Saidakmal02/Uzbek_TTS](https://github.com/Saidakmal02/Uzbek_TTS) («train on Common voice», 6 yulduz), [uzbrainai/vits2_uz](https://github.com/uzbrainai/vits2_uz) (1 yulduz), [jovlievs/sitora-ai](https://github.com/jovlievs/sitora-ai) («10+ hours», MIT, 1 yulduz)
- **Hech biri zero-shot ovoz klonlash emas.** Hammasi bir ovozli VITS/Tacotron.
- **OmniVoice ma'lumot ro'yxatida na USC, na UzbekVoice bor** — ya'ni eng yaxshi o'zbek korpuslari hali ishlatilmagan.

---

## 8. Mavjud o'zbek TTS / ovoz klonlash urinishlari — to'liq inventarizatsiya

### 8.1 Ochiq modellar (HuggingFace)

| Model | Nima | Litsenziya | Zero-shot klonlash? | Sifat |
|---|---|---|---|---|
| [facebook/mms-tts-uzb-script_cyrillic](https://huggingface.co/facebook/mms-tts-uzb-script_cyrillic) | Meta MMS VITS | **CC-BY-NC-4.0** | ❌ Yo'q | Past-o'rtacha. **Faqat kirill** — HF'da lotin varianti **mavjud emas** (API qidiruvi bilan tasdiqlandi) |
| [uzlm/sayro-tts-1.7B](https://huggingface.co/uzlm/sayro-tts-1.7B) | **Qwen3-TTS-12Hz-1.7B-Base** bazasida, UzLM jamoasi (Examy.me, Teamwork.uz). 509 yuklab olish | **`sayro-terms-of-use`** — maxsus litsenziya, deepfake taqiqlangan, **yuklab olish uchun qo'lda tasdiq kerak** | ❌ Yo'q — oldindan belgilangan spikerlar («sayro») + «Happy»/«Neutral» instruksiya teglari | Nashr etilgan metrika yo'q. Ma'lumot: «a balanced mixture of **synthetic audio** and diverse public Uzbek speech datasets» — soat soni ko'rsatilmagan |
| [Abduqayum/uzbek-tts-natural-speech-chatterbox](https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox) | **Chatterbox LoRA finetune**, ~30 s bir spikerli audiokitob. 81 yuklab olish | **MIT** ✅ | ✅ **HA — saqlangan** | Metrika nashr etilmagan, faqat demo audio. **Lekin bu o'zbek uchun eng muhim ochiq zero-shot urinish** |
| [MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic](https://huggingface.co/MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic) va shu muallifning 4 ta varianti | MMS finetune'lari | CC-BY-4.0 (bittasi), qolganlarida ko'rsatilmagan | ❌ | Metrika yo'q |
| davron112/mms-tts-uzb-cyrl-onnx, mirodil/vits-mms-tts-uzb-train, KINDNES-s/mms-tts-uzb-finetuned | MMS ONNX/finetune | ko'rsatilmagan | ❌ | Metrika yo'q |
| [IS2AI/TurkicTTS](https://github.com/IS2AI/TurkicTTS) | Tacotron2 + ParallelWaveGAN, qozoq→o'zbek translit | KazakhTTS2: akademik+tijorat | ❌ | **O'zbek: Q=2.85, I=45%** — past |

### 8.2 Yopiq tijorat mahsulotlari

| Mahsulot | Kim | TTS? | Ovoz klonlash? | Ochiqlik |
|---|---|---|---|---|
| **Aisha Space** ([aisha.group/en/tts-uzbek](https://aisha.group/en/tts-uzbek)) | **AI TECH LLC**, Toshkent | ✅ «purpose-built for Uzbek phonetics», erkak+ayol ovoz, mood control, tezlik | ❌ **Klonlash haqida hech narsa yo'q** | **Yopiq, REST API.** Model/dataset ma'lumoti yo'q. Narx e'lon qilinmagan |
| **Mohir.ai** | Mohir Dev (Toshkent) | — | — | ⚠️ **Tekshirilmadi** — `mohir.ai` DNS timeout berdi, qidiruvda ham sahifa chiqmadi. **O'zbek TTS yoki klonlash borligini tasdiqlay olmadim** |
| **Tahrirchi** (`tahrirchi.uz` → **`tilmoch.ai`** ga 301 redirect) | Tahrirchi | — | — | ⚠️ **Tekshirilmadi** — `tilmoch.ai` HTTP 403 qaytardi. Tahrirchi NLP (imlo tuzatish, LLM) bilan mashhur, speech bo'yicha tasdiq yo'q |
| **UzbekVoice** ([uzbekvoice.ai](https://uzbekvoice.ai/), [corpus.uzbekvoice.ai](https://corpus.uzbekvoice.ai/en-US)) | Hasanov Bakhtiyor + IT Park, UNDP | Dataset to'plash loyihasi, model emas | — | «open source» da'vosi, lekin **aniq litsenziya va yuklab olish sharti e'lon qilinmagan** |
| CAMB.AI | Xalqaro | ✅ o'zbek + 150 til | ✅ bor | **Yopiq, pullik API** |

### 8.3 Xulosa (8-bo'lim)

**O'zbek tili uchun ochiq, litsenziyasi toza, sifati o'lchangan zero-shot ovoz klonlash modeli bugun MAVJUD EMAS.**

Eng yaqin ikki narsa:
1. **OmniVoice** — o'zbek bor, sifat o'lchangan (CER 3.37%), lekin **vaznlar CC-BY-NC**
2. **Abduqayum/uzbek-tts-natural-speech-chatterbox** — MIT, klonlash ishlaydi, lekin **sifat o'lchanmagan va 30 soat bir ovozdan**

Mahalliy yopiq mahsulotlar (Aisha) klonlash taklif qilmaydi. Ya'ni **bu bo'shliq hali ham ochiq** — lekin «noldan til qo'shish» bo'shlig'i emas, «o'lchangan, litsenziyasi toza, mahsulot darajasidagi o'zbek klonlash» bo'shlig'i.

---

## 9. 2026 alifbo islohoti — hoziroq matn frontendi uchun nima qilish kerak

### Tasdiqlangan faktlar

| Sana | Nima bo'ldi | Manba |
|---|---|---|
| **2026-07-07** | Oliy Majlis Qonunchilik palatasi «Lotin yozuviga asoslangan o'zbek alifbosini joriy etish to'g'risida»gi qonunni qabul qildi | [kun.uz](https://kun.uz/news/2026/07/07/ozbekistonda-alifbo-islohoti-boyicha-qonun-qabul-qilindi-4cf07b), [spot.uz](https://www.spot.uz/oz/2026/07/07/alphabet-reform) |
| **2026-09-10** | **Senat ma'qulladi** — 28 harf + 1 apostrof | [gazeta.uz](https://www.gazeta.uz/oz/2026/09/10/uzb-alphabet/) |

### O'zgarishlar

| Eski | Yangi |
|---|---|
| `sh` | **`ş`** |
| `ch` | **`ç`** |
| `gʻ` | **`ğ`** |
| `oʻ` | **`ö`** |
| `ts` (o'zlashma so'zlarda) | **`c`** |
| `ng` | alifbodan **alohida birlik sifatida olib tashlandi** (birikma o'zi imlo qoidalari bilan saqlanadi) |

26 harf + 3 birikma → **28 harf + 1 apostrof**. Maqsad: «bir tovush — bir harf» prinsipiga yaqinlashish.

Islohot sababi rasman **raqamli muhit** bilan bog'langan: «The reform addresses technical difficulties with current characters in digital systems, keyboard input, search engines, **and artificial intelligence applications**». Ya'ni bu islohot aynan bizning muammoni hal qilish uchun qilingan.

### Vaqt jadvali

- **2027-yildan** — 1-sinf darsliklari yangi alifboda chop etiladi
- **2031-yilga qadar** — barcha darsliklar to'liq yangilanadi
- Qonun kuchga kirgunga qadar rasmiylashtirilgan hujjatlar, milliy valyuta, qimmatli qog'ozlar **amal qilishda qoladi**
- Peshlavha, ko'rsatkich, binolardagi yozuvlar — jismoniy eskirishiga yoki belgilangan muddatgacha o'zgarishsiz

Manba: [kun.uz](https://kun.uz/news/2026/07/07/ozbekistonda-alifbo-islohoti-boyicha-qonun-qabul-qilindi-4cf07b), [alfavit.uz/reform](https://alfavit.uz/reform)

### Matn frontendi uchun amaliy xulosa — HOZIR nima qilish kerak

**1. Modelga yangi alifboda o'qitmang. Normalizatsiya qatlami quring.**

Barcha mavjud ma'lumot (USC, Common Voice, FLEURS, FeruzaSpeech, OmniVoice'ning 115 soati) **1995 alifbosida**. Bugun yangi alifboda model o'qitish — o'zini ma'lumotsiz qoldirish.

**2. To'g'ri arxitektura: ikki tomonlama normalizator.**

```
Foydalanuvchi matni (eski YOKI yangi alifbo, YOKI kirill)
        ↓
  [Normalizator]  →  modelning kanonik yozuvi (hozir: 1995 lotin)
        ↓
      Model
```

Bu qatlam **bugundan** quyidagilarni qamrashi kerak:
- `ş → sh`, `ç → ch`, `ğ → gʻ`, `ö → oʻ`, `c → ts` (yangi → eski)
- Apostrof variantlarining barchasini birlashtirish: `ʻ`, `'`, `'`, `´`, `` ` `` → bitta kanonik belgi. **Bu bugunning eng katta real xatolik manbasi**, islohotdan ko'ra muhimroq
- Kirill → lotin (ma'lum F1 0.89 bilan — uzbekvoice-filtered kirillda bo'lgani uchun kerak)
- Sonlar, qisqartmalar, birliklar, chet so'zlar

**3. Model 2031 gacha o'zgartirilmasligi kerak.** Normalizator mapping jadvalini almashtirish — bir soatlik ish. Modelni qayta o'qitish — haftalik ish. Shuning uchun **butun alifbo riskini normalizator qatlamiga joylashtirish** to'g'ri qaror.

**4. Kutilayotgan foyda (muhim).** Islohotdan keyin o'zbek yozuvi turk yozuviga ancha yaqinlashadi (`ş ç ğ ö` — bularning uchtasi turk alifbosida bor). Bu kelajakda **turk tilidan transfer imkoniyatini ochadi** — hozir orfografiya mos kelmagani uchun bu yopiq (3-bo'lim). Ya'ni 2027+ da XTTS-v2/Chatterbox'ning turk komponenti ancha foydaliroq bo'lishi mumkin.

**5. Hali noaniq.** Davlat ma'lumot tizimlarini moslashtirish rejasi e'lon qilingan, lekin **model/dataset re-normalizatsiyasi uchun rasmiy standart yoki referens translit kutubxonasi hali yo'q.** Bu bo'shliq — kichik, foydali ochiq loyiha mavzusi (o'zbek alifbo normalizatori).

---

## 10. Tavsiya etilgan yo'l

### Bosqich 0 — 1 kun: OmniVoice'ni o'zbekda sinash (eng muhim qadam)

```bash
pip install omnivoice
# 10 sekundlik o'zbek referens audio + o'zbek lotin matn + language_id="uz"
```

Nimani o'lchash:
1. Tushunarlilik — **Kotib ASR** (CER 7.0%) bilan sintez qilingan audiodan CER
2. Ovoz o'xshashligi — referens vs sintez SIM-o
3. `oʻ`, `gʻ`, `sh`, `ch`, `ng` aniq to'g'ri talaffuz qilinadimi
4. **Modelning `uz` ma'lumoti lotinda yoki kirillda bo'lganini aniqlash** — ikkalasini ham sinash
5. Sonlar, qisqartmalar, rus/ingliz o'zlashmalari

Bu sinov **hamma narsani hal qiladi.** Agar natija yaxshi bo'lsa, 3-oylik loyiha 1-haftalik loyihaga aylanadi.

### Bosqich 1 — matn frontendi (1 hafta)
Normalizator: apostrof unifikatsiyasi, yangi↔eski alifbo, kirill→lotin, sonlar/qisqartmalar.

### Bosqich 2 — agar sifat yetarli bo'lmasa: LoRA fine-tune (2–4 hafta)
Ma'lumot: USC audiokitob qismi (~10 s, eng toza) + WV-MOS/NISQA filtrlangan USC crowdsourced + filtrlangan Common Voice validated → ~70–110 soat, ~1,500+ spiker. FeruzaSpeech sifat kalibratori sifatida, **baza emas**. `examples/run_finetune_lora.sh`.

Majburiy nazorat: fine-tune'dan oldin/keyin **OOD (o'zbek bo'lmagan) spikerlarda SIM-o** — tushsa, forgetting boshlangan, to'xtatish kerak.

### Bosqich 3 — tijorat kerak bo'lsa: Chatterbox yo'li (parallel)
MIT litsenziyasi uchun. Boshlanish nuqtasi — Abduqayum'ning mavjud o'zbek LoRA'si, uni ko'p spikerli USC bilan kengaytirish. Grapheme tokenizer o'zbek lotinini o'zgarishsiz qamraydi. Ogohlantirish: vocab size shape mismatch muammosi.

### Parallel: UzbekVoice'ning ~1200 soatini ochish
Bu yagona narsa sifatni sakrashga olib keladi. Hasanov Bakhtiyor / IT Park bilan litsenziya va texnik spetsifikatsiya (sample rate, spiker soni) haqida bog'lanish.

### Qilmaslik kerak
- ❌ **XTTS-v2'ga noldan o'zbek qo'shish** — 48–100 soat + tokenizer kengaytirish + CPML litsenziya, va OmniVoice'da bu allaqachon bor
- ❌ **espeak-ng'ga tayanish** — `uz` «status testing», 90 qatorli qoida fayli, ona tili tekshiruvi yo'q
- ❌ **Zonos** — espeak frontendi bor, lekin faqat 5 tilda o'qitilgan, o'zbek akustikasi yo'q
- ❌ **Turkiy transliteratsiya** — TurkicTTS o'lchagan: I=45%, Q=2.85
- ❌ **FeruzaSpeech'ni asosiy o'quv manbai qilish** — bir spiker ko'p ovozli zero-shot'ni buzadi
- ❌ **MegaTTS3** — WaveVAE encoder vaznlari chiqarilmagan, o'z ovozingiz uchun ularning navbatiga so'rov kerak

---

## Tadqiqot cheklovlari (halol ro'yxat)

Tasdiqlay olmadim:
1. **mohir.ai** — DNS timeout; o'zbek TTS/klonlash borligi tekshirilmadi
2. **tilmoch.ai (Tahrirchi)** — HTTP 403; speech mahsulotlari tekshirilmadi
3. **CosyVoice 3**ning to'liq 9 tillik ro'yxati — abstraktda yo'q (lekin v2 bilan bir xil ekani ehtimoli yuqori, ya'ni o'zbek yo'q)
4. **Orpheus multilingual** 7 tilining aniq ro'yxati — README'da berilmagan
5. **USC sample rate** — maqolada ko'rsatilmagan (faqat «WAV»). Telegram bot orqali yig'ilgani uchun OGG/Opus'dan konvertatsiya qilingan bo'lishi mumkin — bu TTS uchun sezilarli yo'qotish. **Datasetni yuklab tekshirish kerak**
6. **USC litsenziyasi** — maqola CC BY 4.0, HF kartada MIT. **Kelishmovchilik, aniqlashtirish kerak**
7. **OmniVoice'ning o'zbek 115 soati qaysi manbadan** — maqolada til-manba xaritasi berilmagan
8. **espeak-ng `uz`ning nashr etilgan xato ro'yxati** — mavjud emas (hech kim hisobot qilmagan)
9. **UzbekVoice**ning spiker soni, sample rate, aniq litsenziyasi — e'lon qilinmagan
10. **Abduqayum'ning o'zbek Chatterbox'i uchun sifat metrikasi** — nashr etilmagan

WebSearch budjeti: **11 / 18** ishlatildi. Qolgan ma'lumot HF/GitHub/arXiv to'g'ridan-to'g'ri fetch orqali olindi. Budjet limitiga yetmadi.

---

## Barcha manbalar

**Zero-shot modellar**
- OmniVoice: [arXiv:2604.00688](https://arxiv.org/abs/2604.00688) · [GitHub](https://github.com/k2-fsa/OmniVoice) · [HF](https://huggingface.co/k2-fsa/OmniVoice) · [languages.md](https://raw.githubusercontent.com/k2-fsa/OmniVoice/main/docs/languages.md) · [LICENSE](https://raw.githubusercontent.com/k2-fsa/OmniVoice/main/LICENSE)
- CosyVoice2: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B · CosyVoice3: https://arxiv.org/abs/2505.17589
- OpenAudio S1-mini: https://huggingface.co/fishaudio/openaudio-s1-mini
- MaskGCT: https://huggingface.co/amphion/MaskGCT
- XTTS-v2: https://huggingface.co/coqui/XTTS-v2 · yangi til: https://github.com/coqui-ai/TTS/discussions/3689 · https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages
- F5-TTS: https://huggingface.co/SWivid/F5-TTS · X-Voice: https://arxiv.org/abs/2605.05611 · Cross-Lingual F5-TTS 2: https://arxiv.org/abs/2609.15184
- IndexTTS: https://github.com/index-tts/index-tts
- Zonos: https://huggingface.co/Zyphra/Zonos-v0.1-transformer
- Chatterbox: https://huggingface.co/ResembleAI/chatterbox · https://github.com/resemble-ai/chatterbox · finetune kit: https://github.com/gokhaneraslan/chatterbox-finetuning
- Orpheus: https://github.com/canopyai/Orpheus-TTS
- Higgs Audio v2: https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base
- VibeVoice: https://huggingface.co/microsoft/VibeVoice-1.5B
- Llasa-3B: https://huggingface.co/HKUSTAudio/Llasa-3B
- Spark-TTS: https://huggingface.co/SparkAudio/Spark-TTS-0.5B
- MegaTTS3: https://huggingface.co/ByteDance/MegaTTS3
- Parler-TTS: https://huggingface.co/parler-tts/parler-tts-large-v1
- MMS-TTS: https://huggingface.co/facebook/mms-tts-uzb-script_cyrillic

**Turkiy transfer**
- TurkicTTS: https://github.com/IS2AI/TurkicTTS · https://arxiv.org/abs/2305.15749

**espeak-ng**
- uz voice fayli: https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/espeak-ng-data/lang/trk/uz
- uz_rules: https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/dictsource/uz_rules
- voices.md (status ta'rifi): https://github.com/espeak-ng/espeak-ng/blob/master/docs/voices.md
- languages.md: https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/docs/languages.md

**Catastrophic forgetting / bir spikerli fine-tune**
- PEFT for Low-Resource TTS (INTERSPEECH 2025): https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf
- Bayesian PEFT: https://arxiv.org/pdf/2402.12220
- Residual Adapters: https://lacuna.tiptreesystems.com/work/residual-adapters-for-few-shot-text-to-speech-speaker-adaptation/wrk_3885e6ec7a8dc1a03d280b298698b01e
- Align2Speak: https://arxiv.org/abs/2509.21718
- Chatterbox low-resource amaliyoti (slovak): https://dev.to/pekisko/fine-tuning-chatterbox-on-a-low-resource-language-7-things-that-mattered-13e1

**O'zbek ma'lumotlari**
- USC: https://arxiv.org/abs/2107.14419 · https://github.com/IS2AI/Uzbek_ASR · https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus
- Common Voice uz 25.0: https://mozilladatacollective.com/datasets/cmn29fa4y0150o1073fqww73p
- uzbekvoice-filtered: https://huggingface.co/datasets/DavronSherbaev/uzbekvoice-filtered
- UzbekVoice: https://uzbekvoice.ai/ · https://corpus.uzbekvoice.ai/en-US
- FLEURS: https://arxiv.org/abs/2205.12446 · https://huggingface.co/datasets/google/xtreme_s
- Common Voice → TTS filtrlash: https://arxiv.org/abs/2210.06370 · https://www.isca-archive.org/iberspeech_2024/giraldo24_iberspeech.pdf

**O'zbek TTS urinishlari**
- sayro-tts: https://huggingface.co/uzlm/sayro-tts-1.7B
- O'zbek Chatterbox: https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox
- Aisha: https://aisha.group/en/tts-uzbek
- Kichik loyihalar: https://github.com/Saidakmal02/Uzbek_TTS · https://github.com/uzbrainai/vits2_uz · https://github.com/jovlievs/sitora-ai

**Alifbo islohoti**
- Senat (2026-09-10): https://www.gazeta.uz/oz/2026/09/10/uzb-alphabet/
- Qonun (2026-07-07): https://kun.uz/news/2026/07/07/ozbekistonda-alifbo-islohoti-boyicha-qonun-qabul-qilindi-4cf07b · https://www.spot.uz/oz/2026/07/07/alphabet-reform
- Klaviatura: https://www.gazeta.uz/oz/2026/09/10/keyboard-uzb--translation-2026-09-10-13-10-21/
- Alfavit: https://alfavit.uz/reform

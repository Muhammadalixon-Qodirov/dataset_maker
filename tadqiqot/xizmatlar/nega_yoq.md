# Nega hali hech kim avtomatik TTS dataset tayyorlashni mahsulot/xizmatga aylantirmagan?

**Tadqiqot sanasi:** 2026-10-01
**Mavzu:** Biznes, huquqiy va strukturaviy to'siqlar — va haqiqiy imkoniyat bormi?

> **Diqqat:** Bu hujjat tijorat ovoz-klonlash xizmatlari katalogi EMAS va tadqiqot pipeline'larining bosqichma-bosqich tavsifi ham emas. Bu — *nega bu bozor shakllanmaganining* dalillar bilan tahlili.

---

## 0. Qisqa javob (TL;DR)

Besh sabab bir-birini kuchaytiradi:

1. **Avtomatlashtiriladigan qism deyarli bepul.** ASR API bugun 1 soat audioni $0.15–0.36 ga transkripsiya qiladi ([AssemblyAI](https://www.assemblyai.com/pricing), [Whisper API narxi](https://diyai.io/ai-tools/speech-to-text/openai-whisper-api-pricing-2026/)). Forced alignment, segmentatsiya, SNR filtri — hammasi ochiq kodli. Hatto **to'liq pipeline allaqachon nashr etilgan va bepul**: Emilia-Pipe "birinchi ochiq kodli preprocessing pipeline" sifatida 101k+ soat datasetni ishlab chiqardi ([arXiv:2407.05361](https://arxiv.org/abs/2407.05361)). Siz sotmoqchi bo'lgan narsa — allaqachon GitHub'da.
2. **Avtomatlashtirilmaydigan qism esa juda qimmat.** Bambara tilida ASR chiqishini tuzatish uchun **1 soat audioga 30 odam-soat** (laboratoriya), **36 odam-soat** (dala sharoitida) ketgan ([arXiv:2510.12781](https://arxiv.org/abs/2510.12781)). Ya'ni avtomatlashtirish narxning 1%ini oladi, qolgan 99% odamda qoladi.
3. **Katta laboratoriyalarga vendor kerak emas.** Whisper 680,000 soat web'dan scrape qilingan audioda o'qitilgan ([arXiv:2212.04356](https://arxiv.org/abs/2212.04356)), Google USM — 12 million soat ([arXiv:2303.01037](https://arxiv.org/pdf/2303.01037)). Eng katta tijorat vendor paketi — 3,125 soat, €500,000 ([Defined.ai DAP](https://defined.ai/press-room/defined-ai-launches-data-access-plan-dap-allowing-access-to-speech-datasets-at-affordable-prices)). Bu Whisper korpusining **0.46%**i.
4. **Ovoz — biometrik ma'lumot, va bu yuridik mina maydoni.** Illinois BIPA bo'yicha har bir qoidabuzarlik uchun $1,000 (ehtiyotsizlik) / $5,000 (qasddan) ([Zaluda v. Apple](https://www.sgtlaw.com/cases/apple-siri-bipa-litigation)). 2026-yil may oyida audiokitob o'quvchilari va jurnalistlar Google, Apple, Microsoft, Meta, ElevenLabs, NVIDIA va boshqalarga qarshi 9 ta class action ariza berdi — aynan **"long-form, single-speaker, studio-quality" audio scrape qilingani** uchun ([CBS Chicago](https://www.cbsnews.com/chicago/news/illinois-class-action-lawsuits-voices-stolen-train-ai-bipa/)). Sizning g'oyangiz ("uzun matn + uzun audio") aynan shu tavsifga tushadi.
5. **Urinib ko'rganlar pivot qildi yoki yopildi.** cielo24 — 2012-yildan 70 xodim, VC puli, 10 yillik time-aligned speech dataset — yopildi; ma'lumot aktivlari "mid-six figures" ($500k–900k) ga sotildi ([SimpleClosure](https://simpleclosure.com/customer-stories/cielo24/)). 15.ai — 2022-yil sentabrda AI/mualliflik huquqi muammolari tufayli o'chdi ([Wikipedia](https://en.wikipedia.org/wiki/15.ai)).

**Lekin niche bor.** U "dataset tayyorlash SaaS" shaklida emas — **(a)** kam resursli tillar uchun grant-asosli dataset ishlab chiqarish (Gates Foundation Masakhane'ga $4,998,785 berdi), **(b)** on-prem/self-hosted tooling (audio tashqariga chiqmasligi kerak bo'lgan holatlar), **(c)** huquqiy tozalash + alignment birgalikda (podcast/audiokitob litsenziya vositachiligi). Batafsil — 7-bo'limda.

---

## 1. Huquqiy to'siq: ovoz endi biometrik ma'lumot

### 1.1 Illinois BIPA — eng xavfli qonun

**BIPA** (Biometric Information Privacy Act, 740 ILCS 14/) 2008-yilda qabul qilingan. Talablari: yozma xabardor qilish, yozma rozilik, saqlash siyosatini e'lon qilish.

- **Zarar:** har bir qoidabuzarlik uchun **$1,000** (ehtiyotsizlik) yoki **$5,000** (qasddan/beparvolik) ([sgtlaw.com](https://www.sgtlaw.com/cases/apple-siri-bipa-litigation)).
- **Rosenbach v. Six Flags Entertainment Corp. (2019, Illinois Oliy sudi):** da'vogar *real zarar* ko'rsatishi shart emas — "qonunning oddiy buzilishining o'zi zarar olish uchun yetarli" ([Wikipedia: BIPA](https://en.wikipedia.org/wiki/Biometric_Information_Privacy_Act)).

**Ovoz bilan bog'liq haqiqiy ishlar:**

| Ish | Sud | Sana | Natija/holat |
|---|---|---|---|
| **Whole Foods** (voiceprint) | Cook County Circuit Court, Chancery Div. | Dastlabki tasdiq **2023-01-03** | ~**$300,000** kelishuv. Da'vogar advokati: "menga ma'lum bo'lishicha, bu ovoz biometrikasiga oid birinchi BIPA kelishuvi" ([Hunton](https://www.hunton.com/privacy-and-cybersecurity-law-blog/whole-foods-settles-bipa-voiceprint-class-action0)) |
| **Zaluda v. Apple Inc.** (Siri voiceprint) | Cook County Circuit Court, hakim Michael T. Mullen | Class sertifikatsiya **2026-01-29** | Sinfga ajratildi, discovery deyarli tugagan, sudga yo'l oldi ([sgtlaw](https://www.sgtlaw.com/cases/apple-siri-bipa-litigation)) |
| **McGoveran v. Amazon Web Services** | 3rd Circuit (AQSh apellyatsiya sudi) | Sana manbada ko'rsatilmagan | **Da'vo rad etildi** — asosiy harakatlar Illinois tashqarisida bo'lgan; Amazon voiceprint'ni o'zi yig'magan (Pindrop yig'gan) ([FindLaw](https://www.findlaw.com/legalblogs/third-circuit/is-your-voiceprint-protected-amazons-biometric-privacy-battle-in-illinois/)) |
| **John Hancock** (Pindrop + Amazon Connect) | Illinois | — | Hukm BIPA voiceprint da'volarining **cheklovlarini** ko'rsatdi; sud voiceprint "identifikatsiya qiluvchi" biometrik ma'lumotmi degan savolni muhim deb topdi ([TopClassActions](https://topclassactions.com/lawsuit-settlements/lawsuit-news/john-hancock-voiceprint-ruling-highlights-limits-of-illinois-biometric-privacy-law/) — batafsil matnni olishga muvaffaq bo'lmadim, faqat sarlavha va qidiruv tavsifi) |

**🔴 Eng muhim ish — 2026-yil may:** Chicago federal sudida bir hafta ichida **9 ta class action** ariza berildi.

- **Da'vogarlar:** Carol Marin (NBC 5, pensiyada), Phil Rogers (NBC 5), Robin Amer (jurnalist), **Lindsay Dorcus** va **Victoria Nassif** (audiokitob o'quvchilari / voice actor), Yohance Lacour va Alison Flowers (podcaster) ([CBS Chicago](https://www.cbsnews.com/chicago/news/illinois-class-action-lawsuits-voices-stolen-train-ai-bipa/)).
- **Javobgarlar:** Amazon, Adobe, Google, Alphabet, Apple, Microsoft, Samsung, Meta (Facebook), **ElevenLabs**, NVIDIA.
- **Ariza sanasi:** 2026-05-19 ([The Bookseller](https://www.thebookseller.com/news/us-audiobook-narrators-sue-google-for-using-their-voices-in-ai-training) — paywall, faqat sarlavha va sana olindi).
- **Google ishi (N.D. Ill.):** Google Assistant va Gemini Live nishonda. Da'vogarlar aytishicha, ularning yozuvlari Google'ning o'zi "optimal training audio" deb belgilagan mezonga to'g'ri keladi: **"long-form, single-speaker, studio-quality, and professionally produced audio"** ([ciplawyer](https://www.ciplawyer.com/articles/159009.html), [Playwire tahlili](https://www.playwire.com/blog/googles-voice-ai-lawsuit-is-a-warning-for-every-content-creator)). Da'volar: right of publicity + BIPA.

> **Bu sizning loyihangiz uchun nimani anglatadi:** "uzun matn + uzun audio → TTS dataset" formulasi aynan sud ishida aybnoma qilib keltirilgan ma'lumot profilini tasvirlaydi. Agar siz bu pipeline'ni *xizmat* sifatida sotsangiz, mijozlaringiz scrape qilgan audiosi uchun siz ham zanjirning bir bo'g'ini bo'lasiz.

### 1.2 GDPR — ovoz Article 9 "maxsus toifa" bo'lib qolishi mumkin

- Ovoz ma'lumoti **avtomatik** Article 9 ga tushmaydi. EDPB yondashuvi: hal qiluvchi omil — *nazoratchining maqsadi*, ya'ni maxsus toifa ma'lumotini chiqarib olish niyati bormi ([Janvier Parewyck tahlili, EDPB yondashuviga asoslangan](https://www.linkedin.com/pulse/when-voice-special-category-personal-data-under-gdpr-janvier-parewyck)).
- Lekin **voiceprint (ovoz modeli) yaratish** — ya'ni enrolment bosqichi — shaxsni yagona identifikatsiya qilish maqsadida biometrik ma'lumot qayta ishlash hisoblanadi va **Article 9 GDPR**ga tushadi (yuqoridagi tahlil EDPB'ning virtual voice assistant'lar bo'yicha ko'rsatmalariga ishora qiladi).
- Oqibati: **explicit consent** (Art. 9(2)(a)) + **DPIA** majburiy ([dilr.ai](https://www.dilr.ai/blog/ai-voice-biometric-data-security-enterprise)).
- **Purpose limitation muammosi:** dataset uchun yig'ilgan rozilik keyinchalik model o'qitish uchun avtomatik kengaymaydi. Har bir yangi maqsad — yangi huquqiy asos.

⚠️ *Bu bo'limda birlamchi EDPB hujjatini to'g'ridan-to'g'ri olib ololmadim (faqat ikkilamchi tahlillar). EDPB Guidelines 02/2021 (virtual voice assistants) ni mustaqil tekshirish tavsiya etiladi.*

### 1.3 EU AI Act — 50-modda, sintetik audio shaffofligi

- **Qo'llanish boshlanishi: 2026-yil 2-avgust** ([artificialintelligenceact.eu](https://artificialintelligenceact.eu/transparency-rules-article-50/), [hard2bit](https://hard2bit.com/en/blog/ai-act-article-50-ai-transparency-chatbots-deepfakes/)).
- **50(2):** sintetik audio/rasm/video/matn ishlab chiqaruvchi provayderlar chiqishni **machine-readable formatda belgilashi** va "artificially generated or manipulated" sifatida **aniqlanadigan** qilishi shart. Yechimlar "effective, interoperable, robust and reliable as far as this is technically feasible" bo'lishi kerak ([Article 50 matni](https://artificialintelligenceact.eu/article/50/)).
- **50(4):** deepfake ishlab chiqaruvchi *deployer*lar kontent sun'iy yaratilganini **oshkor qilishi** shart (huquq-tartibot va badiiy/satirik/fantastik asarlar uchun istisnolar).
- Amaliy talqin: deepfake audio boshida tushunarli ovozli disclaimer; 30 soniyadan uzun kontentda takrorlanishi ([truescreen](https://truescreen.io/insights/ai-act-article-50-labelling-synthetic-content-august-2026/)).
- **Jarima:** €15 million gacha yoki global yillik aylanmaning 3% — qaysi biri yuqori bo'lsa.
- Yevropa Komissiyasi 2026-yil yanvarda **AI Labelling and Transparency bo'yicha Code of Practice loyihasini** chiqardi ([Jones Day](https://www.jonesday.com/en/insights/2026/01/european-commission-publishes-draft-code-of-practice-on-ai-labelling-and-transparency)).

### 1.4 AQSh shtat qonunlari — raqamli nusxa (digital replica)

**Tennessee ELVIS Act** (Ensuring Likeness, Voice, and Image Security Act; Tennessee Public Chapter 588):
- AI ovoz klonlash uchun maxsus yozilgan **birinchi AQSh qonuni**.
- **Ovozni** shaxsiy mulk huquqi sifatida himoya qiladi — ham haqiqiy yozuv, ham "readily identifiable" simulyatsiya ([identity.org](https://www.identity.org/what-is-tennessees-elvis-act/), [spirelight](https://www.spirelight.ai/guides/voice-cloning-laws)).
- **🔴 Siz uchun eng xavfli qismi:** qonun "primary purpose or function"i ruxsatsiz raqamli nusxa yaratish bo'lgan **algoritm, texnologiya, dasturiy ta'minot va vositalarga** ham javobgarlik yuklaydi. Ya'ni *tool* sotuvchi ham javobgar.
- **Kuchga kirish sanasi — manbalar qarama-qarshi:** [identity.org](https://www.identity.org/what-is-tennessees-elvis-act/) — **2024-07-21**; [spirelight.ai](https://www.spirelight.ai/guides/voice-cloning-laws) — **2024-07-01**. Ikkalasini ham keltiraman; rasmiy matndan tekshirish kerak.

**California:**
- **AB 2602** — shaxsiy/professional xizmat shartnomalarida raqamli nusxa haqidagi bandlar **"reasonable specificity" bilan ko'zda tutilgan foydalanishlarni tasvirlamasa, bajarilishi mumkin emas** ([InsideGlobalTech](https://www.insideglobaltech.com/2024/09/11/california-passes-digital-replica-legislation-as-congress-considers-federal-approach/), [Fenwick](https://www.fenwick.com/insights/publications/californias-new-ai-laws-limit-uses-of-digital-likeness)).
- **AB 1836** — vafot etgan shaxsning ovozi/qiyofasining raqamli nusxasini **merosxo'rlar roziligisiz** ishlab chiqarish/tarqatishni taqiqlaydi ([Proskauer](https://www.proskauer.com/blog/california-enacts-generative-ai-law-addressing-digital-replicas-of-performers)).
- Ikkisi 2024-yil sentabrda imzolangan. ⚠️ **Aniq kuchga kirish sanasi men tekshirgan manbalarda ko'rsatilmagan — ma'lumot topilmadi.**

**New York:** NY Civil Rights Law **§§50–51** (right of publicity) — *Lehrman v. Lovo* ishida ovozni o'zlashtirish da'vosining asosi bo'ldi (quyida).

### 1.5 SAG-AFTRA — kasaba uyushmasi shartnomalari

**2025 Interactive Media Agreement (video o'yinlar)** — 2025-yil iyulda ratifikatsiya qilindi, **95.04%** ovoz bilan; ~11 oylik strayk **2025-07-09**da tugadi ([Deadline](https://deadline.com/2025/07/sag-aftra-video-game-agreement-approved-strike-1236453741/), [Wikipedia](https://en.wikipedia.org/wiki/2024%E2%80%932025_SAG-AFTRA_video_game_strike)).

Talablari ([Davis+Gilbert](https://www.dglaw.com/sag-aftras-new-video-game-agreement/), [aitrace](https://www.aitrace.org/company/sag-aftra/practice/1f131e05-3c62-495a-9596-4aa1930e8db2)):
- **Alohida, yozma, aniq va ko'zga tashlanadigan, "reasonably specific" rozilik** — raqamli nusxa yaratishdan *va* ishlatishdan oldin.
- Rozilikda oshkor qilinishi shart: o'yin oldingi IP'dan foydalanadimi, ijrochi mavjud rolni qaytaradimi, real-time generation bo'ladimi.
- **Uch toifa:** Digital Replica, Independently Created Digital Replica (ICDR), Real-Time Generation.
- **Trigger: "objectively identifiable"** — natija aniq bir ijrochi sifatida obyektiv tanib olinadigan bo'lsa, himoya ishga tushadi.
- **Strike suspension clause:** ijrochilar strayk davomida yangi AI material yaratish uchun rozilikni bekor qilishi mumkin — AI straykbreyker bo'lib ishlatilmasligi uchun.
- **Kompensatsiya:** umumiy ish haqi 15.17% kompaund o'sish + 2025, 2026, 2027 noyabrda qo'shimcha 3%. Digital Replica va Real-Time Generation uchun alohida minimal stavkalar belgilangan (aniq raqamlar men olgan manbalarda yo'q — **ma'lumot topilmadi**).

**Boshqa SAG-AFTRA AI kelishuvlari:**
- **Replica Studios** (2024-yil yanvar, CES'da e'lon qilindi) — ixtiyoriy; a'zolar AI ovoz nusxasini yaratib litsenziyalashi mumkin; informed consent + har foydalanish uchun minimal scale kompensatsiya ([SAG-AFTRA](https://www.sagaftra.org/sag-aftra-and-replica-studios-introduce-groundbreaking-ai-voice-agreement-ces)).
- **Television Animation Agreement** (2024-yil mart) — AI ijrochi ismi bilan prompt qilinganda rozilik; sintetik ovoz yaratish uchun xabardor qilish va muzokara.
- **Dynamic AI Audio Commercials Waiver** (2024-yil iyun) — har bosqichda informed consent.

### 1.6 Haqiqiy sud ishlari — case law

**🔑 Lehrman & Sage v. Lovo, Inc.** — 790 F.Supp.3d 348 (S.D.N.Y. 2025), hukm **2025-07-10**:

Voice actor'lar Paul Lehrman va Linnea Sage Lovo'ga qarshi. 2019–2020 yillarda Fiverr orqali ish oldilar; buyurtmachilar keyinchalik Lovo xodimlari bo'lib chiqdi ([CBS News](https://www.cbsnews.com/news/two-voice-actors-sue-ai-company-lovo/)).

| Da'vo | Natija |
|---|---|
| Breach of contract | **Davom etadi** |
| NY Civil Rights Law §§50–51 (right of publicity) | **Davom etadi** |
| NY GBL §§349–350 (consumer protection) | **Davom etadi** |
| Copyright — Sage'ning haqiqiy ovoz yozuvi investor prezentatsiyasida ishlatilgani | **Davom etadi** (bitta da'vo) |
| Copyright — AI training va yaratilgan ovoz klonlari | **Rad etildi** |
| Lanham Act (false association, false advertising) | **Rad etildi** |
| Common law fraud, unjust enrichment, conversion, unfair competition | **Rad etildi** |

**Sudning asosiy mulohazasi:** *"copyright law does not protect against imitation or simulation of a voice, only against direct copying of a fixed recording"* ([Loeb & Loeb](https://www.loeb.com/en/insights/publications/2025/07/lehrman-v-lovo-inc), [Crowell](https://www.crowell.com/en/insights/client-alerts/fundamental-copyright-principles-underscored-in-ai-context-voice-attributes-are-not-protectable)).

> **Bu nima degani:** mualliflik huquqi sizni qutqarmaydi — lekin himoya ham qilmaydi. Ovoz atributlari copyright bilan himoyalanmaydi, demak **right of publicity va shartnoma** huquqi asosiy jang maydoni. Va bu huquqlar shtatga ko'ra o'zgaradi, federal preemption yo'q. 50 shtat × noaniq qoida = compliance do'zaxi.

**Scarlett Johansson / OpenAI "Sky" epizodi (2024-yil may):**
- Sam Altman 2023-yil sentabrda Johansson'ga ChatGPT uchun ovoz berishni taklif qildi; u rad etdi.
- 2024-yil mayda "Sky" ovozi chiqdi. Johansson advokat jalb qildi; advokatlari Altman'ga **ikki xat** yubordi, "Sky" ovozi qanday yaratilganini aniq so'radi.
- OpenAI "Sky"ni olib tashlashga rozi bo'ldi. Altman 2024-05-20 bayonoti: *"The voice of Sky is not Scarlett Johansson's, and it was never intended to resemble hers. We cast the voice actor behind Sky's voice before any outreach to Ms. Johansson."* ([CNBC](https://www.cnbc.com/2024/05/20/scarlett-johansson-says-openai-ripped-off-her-voice-.html), [TechCrunch](https://techcrunch.com/2024/05/20/openai-to-remove-chatgpts-scarlett-johansson-like-voice), [Today](https://www.today.com/popculture/news/scarlett-johansson-openai-voice-sky-rcna153188)).
- **Da'vo arizasi berilganiga dalil topilmadi — ma'lumot topilmadi.** Lekin xulosa aniq: *sud ishi bo'lmasa ham, reputatsion bosim ovoz mahsulotini bozordan olib tashlashga majbur qildi.*

### 📋 JADVAL: Huquqiy cheklovlar

| Qonun / kelishuv | Yurisdiksiya | Nimani talab qiladi | Holati / sana |
|---|---|---|---|
| **BIPA** (740 ILCS 14/) | Illinois, AQSh | Yozma xabardorlik, yozma rozilik, saqlash siyosati. Zarar: $1,000 / $5,000 har buzilish uchun. Real zarar shart emas (*Rosenbach*, 2019) | **Kuchda, 2008-yildan.** 2026-05 da 9 ta yangi class action (Google, Apple, Meta, ElevenLabs, NVIDIA va b.) |
| **GDPR Art. 9** | YeI/YeIH | Voiceprint = maxsus toifa biometrik ma'lumot → **explicit consent** + **DPIA**. Purpose limitation: yangi maqsad = yangi asos | Kuchda, 2018-yildan |
| **EU AI Act Art. 50** | YeI | Sintetik audio **machine-readable** belgilanishi va aniqlanadigan bo'lishi; deepfake oshkor qilinishi. Jarima €15M / 3% aylanma | **2026-08-02 dan qo'llanadi.** Code of Practice loyihasi 2026-01 |
| **ELVIS Act** (TN PC 588) | Tennessee, AQSh | Ovoz = shaxsiy mulk huquqi; simulyatsiya ham qoplanadi. **"Primary purpose" ruxsatsiz nusxa yaratish bo'lgan tool/software ham javobgar** | Kuchda. Sana: **2024-07-21** ([identity.org](https://www.identity.org/what-is-tennessees-elvis-act/)) yoki **2024-07-01** ([spirelight](https://www.spirelight.ai/guides/voice-cloning-laws)) — manbalar qarama-qarshi |
| **AB 2602** | California, AQSh | Shartnomada raqamli nusxa bandlari "reasonable specificity" bilan foydalanishni tasvirlamasa — **bajarilmaydi** | 2024-09 imzolangan. Kuchga kirish sanasi: **ma'lumot topilmadi** |
| **AB 1836** | California, AQSh | Vafot etgan shaxs ovozining raqamli nusxasi uchun merosxo'rlar roziligi shart | 2024-09 imzolangan. Kuchga kirish sanasi: **ma'lumot topilmadi** |
| **NY Civil Rights Law §§50–51** | New York, AQSh | Ovozni o'zlashtirish right of publicity buzilishi; *Lehrman v. Lovo*da ishlatildi | Kuchda; *Lehrman* da'vosi davom etadi (2025-07-10 hukmi) |
| **SAG-AFTRA 2025 IMA** | AQSh, video o'yinlar | Alohida yozma, aniq, "reasonably specific" rozilik; oshkor qilish; "objectively identifiable" trigger; strike suspension | **Ratifikatsiya 2025-07**, 95.04% |
| **EU DSM Art. 4 (TDM)** | YeI | Tijorat TDM ruxsat — **lekin huquq egasi machine-readable opt-out qilsa, yo'q** | Kuchda (Direktiva 2019/790) |
| **Japan Copyright Act Art. 30-4** | Yaponiya | "Non-enjoyment" maqsadlar uchun keng TDM istisnosi | Kuchda |
| **UK TDM** | Buyuk Britaniya | YeI uslubidagi opt-out istisnosi **taklif qilingan** | Konsultatsiya **2024-12-17** – **2025-02-25**. Hukumat javobi: manbalarimda yakunlanmagan |

---

## 2. Manba materialining mualliflik huquqi

Sizning g'oyangiz audiokitob/kitoblardan foydalanadi. Bu yerda uch alohida huquq qatlami bor va ular bir-biriga mos kelmaydi.

### 2.1 Public domain yo'li — toza, lekin kichik

**LibriVox** ([librivox.org/pages/public-domain](https://librivox.org/pages/public-domain/)):
- *"all our recordings are public domain (definitely in the USA, and maybe in your country as well)"*
- Foydalanuvchilar *"can do anything they like"* — sotish, eshittirish, reklamada ishlatish, remiks qilish.
- Atribut shart emas.
- **Lekin:** sahifa *"we can only record texts published before 1923"* deydi (sahifa eskirgan — AQSh PD chizig'i har yil suriladi). Ya'ni **faqat 100 yildan oshgan matnlar**.
- **🔴 Siz uchun muammo:** eski adabiy til, zamonaviy so'zlashuv yo'q, aksent/sifat juda har xil, va **o'zbek tili uchun LibriVox'da deyarli hech narsa yo'q**.
- Geografik ogohlantirish: AQShda PD bo'lgani boshqa davlatda PD ekanini kafolatlamaydi.

**Project Gutenberg** ([gutenberg.org/policy/permission.html](https://www.gutenberg.org/policy/permission.html)):
- Ko'pchilik kitob AQShda PD — har qanday tijorat foydalanish, qayta nashr, derivativ asar ruxsat.
- **Lekin:** "Project Gutenberg" **registratsiya qilingan savdo belgisi**. Tijorat loyihalar savdo belgisi nomidan foyda ko'rsa, royalti to'lashi kerak (proporsional: 10 manbadan 1 tasi PG bo'lsa — standart royaltining 1/10).
- **AI o'qitish haqida sahifada hech qanday bayonot yo'q** — ya'ni noaniqlik.

### 2.2 Tijorat audiokitoblar — eng katta mina

- **🔴 Google ishi aynan shu haqida.** Audiokitob o'quvchilari (Dorcus, Nassif) Google'ni audiokitob audiosini scrape qilib AI o'qitgani uchun sudga berdi (2026-05, N.D. Ill.). Audiokitobda kamida **uch huquq egasi** bor: matn muallifi/nashriyot, ijrochi (narrator), produser/platforma.
- Bir tahlilchi buni shunday ifodalaydi: audio matn va rasmda yo'q **uch qatlam** huquqiy riskni olib yuradi — biometrik shaxs, copyright chalkashligi (eshittirish litsenziyasi AI o'qitishga kengaymaydi), va rozilik murakkabligi (bir yozuvda ko'p huquq egasi) ([Box Commons](https://boxcommons.org/blog/audio-data-liability-frontier/)).
- Bozor bu muammoni **litsenziyalash** bilan hal qilishga urinmoqda: Bookwire aytishicha, tex-kompaniyalar audiokitob kontentini AI o'qitish uchun litsenziyalashni so'rab murojaat qilmoqda — ayniqsa **asosiy e'tibor qaratilmagan tillar, masalan portugal tili** uchun ([Publishers Weekly](https://www.publishersweekly.com/pw/by-topic/digital/content-and-e-books/article/97387-the-audiobook-business-learns-to-embrace-ai.html)). Storytel mashhur narratorlarning **litsenziyalangan** klon ovozlarini ishlatadi (stavkalar oshkor qilinmagan).

### 2.3 TDM (text-and-data-mining) istisnolari — yetarli emas

| Yurisdiksiya | Qoida | Cheklov |
|---|---|---|
| **YeI DSM Art. 3** | Tadqiqot tashkilotlari uchun TDM, opt-out yo'q | Faqat tadqiqot muassasalari — startup emas |
| **YeI DSM Art. 4** | **Har qanday maqsad**, tijorat ham — ammo huquq egasi machine-readable tarzda huquqini saqlab qolsa (opt-out), ishlamaydi ([Wolters Kluwer](https://legalblogs.wolterskluwer.com/copyright-blog/the-new-copyright-directive-text-and-data-mining-articles-3-and-4/), [Creative Commons bayonoti](https://creativecommons.org/wp-content/uploads/2021/12/CC-Statement-on-the-TDM-Exception-Art-4-DSM-Final.pdf)) | Buyuk Britaniya hukumatining o'zi tan oladi: opt-out mexanizmi **"amalda qanday ishlashi noaniq"** |
| **Yaponiya Art. 30-4** | "Non-enjoyment" maqsadlar uchun keng istisno; generativ AI uchun eng liberal rejim deb baholanadi ([Dermawan, JWIP 2024](https://onlinelibrary.wiley.com/doi/full/10.1111/jwip.12285)) | Faqat Yaponiya hududida |
| **Buyuk Britaniya** | Opt-out'siz, **barcha maqsadlar** uchun TDM istisnosi taklif qilingan; keyin YeI uslubidagi opt-out variantiga o'tildi | Konsultatsiya 2024-12-17 – 2025-02-25 ([DLA Piper](https://www.dlapiper.com/en-us/insights/blogs/mse-today/2025/training-ai-models-uk-government-proposes-eu-style-opt-out-copyright-exception)). Yakuniy qoida: **ma'lumot topilmadi** |

### 🔴 Eng muhim strukturaviy nuqta

**TDM istisnolari faqat mualliflik huquqini qoplaydi. Ular biometrik va publicity huquqlarini QOPLAMAYDI.**

Ya'ni siz Yaponiyada mukammal TDM pozitsiyasiga ega bo'lishingiz mumkin va shu bilan birga Illinoisda BIPA bo'yicha, Tennesseeda ELVIS Act bo'yicha, New Yorkda §§50–51 bo'yicha javobgar bo'lishingiz mumkin — chunki bu huquqlar mualliflik huquqidan **mustaqil**. *Lehrman v. Lovo* ham aynan shuni ko'rsatdi: copyright da'volari rad etildi, right of publicity da'volari davom etdi.

Bu — bir mahsulot bilan "huquqiy toza dataset" sotishni deyarli imkonsiz qiladi. Har bir mijoz, har bir til, har bir ovoz uchun alohida chain-of-consent kerak.

---

## 3. Iqtisod: kim speech data uchun to'laydi va qancha?

### 3.1 Vendor narxlari (e'lon qilingan)

**Defined.ai Data Access Plan** (press-relize **2024-09-26**):

| Paket | Soat | Narx | Soatiga (mening hisobim) |
|---|---|---|---|
| Standard | 315 | €50,000 | €158.73 |
| Premium | 625 | €100,000 | €160.00 |
| Ultimate | 1,565 | €250,000 | €159.74 |
| Elite | 3,125 | €500,000 | €160.00 |

Chegirmalar "standart narxga nisbatan 30%–55%" deb aytilgan ([Defined.ai](https://defined.ai/press-room/defined-ai-launches-data-access-plan-dap-allowing-access-to-speech-datasets-at-affordable-prices)). **Standart (chegirmasiz) soat narxi e'lon qilinmagan.** Mening hisobim: 30% chegirma = list €227/soat; 55% chegirma = list €356/soat. Bu — hisob, vendor bayonoti emas.

**Boshqa manba** ([speechdata.ai qo'llanmasi](https://www.speechdata.ai/guides/how-to-buy-speech-data)):

| Tur | Soat narxi |
|---|---|
| Off-the-shelf, asosiy tillar | **$50–100** |
| Off-the-shelf, kam resursli tillar | **$60–150** |
| SpeechData.ai katalogi (60 til, 500–2,000 soat/dataset) | **$60–95** |
| **Custom collection** (buyurtma yig'ish) | **$150–400+** |

Custom narxni oshiradigan omillar: demografik talablar, skriptlangan senariylar, eksklyuzivlik, muddat bosimi.

**Odam transkripsiyasi:** $1.50–$3.00 / audio-daqiqa = **$90–180 / audio-soat** ([Notta](https://www.notta.ai/en/blog/transcription-rate), [Sonix](https://sonix.ai/how-much-does-transcription-cost)). Mehnat 1 soat audio uchun 3–6 soat; 5+ ma'ruzachi bo'lsa 8 soatgacha; ba'zi hollarda audio uzunligidan 14× ko'p.

### 3.2 Avtomatlashtirishning narxi

| Xizmat | Soat narxi |
|---|---|
| AssemblyAI Universal-2 | **$0.15** |
| AssemblyAI Universal-3.5 Pro | **$0.21** |
| OpenAI Whisper API | **$0.36** ($0.006/daq) |

Manbalar: [AssemblyAI pricing](https://www.assemblyai.com/pricing), [diyai.io](https://diyai.io/ai-tools/speech-to-text/openai-whisper-api-pricing-2026/).

### 3.3 🔴 Bu taqqoslash biznes modelni o'ldiradi

```
Scraping + ASR:            ~$0.36 / soat
Vendor OTS dataset:        ~$60–160 / soat     →  ~170–440×
Odam transkripsiyasi:      ~$90–180 / soat     →  ~250–500×
Custom collection:         ~$150–400 / soat    →  ~420–1100×
```

**Nega frontier lab tayyor datasetni sotib oladi, scrape qilish o'rniga?** Faqat uch holatda:

1. **Scraping huquqiy riski juda yuqori bo'lganda** — bu bugun o'zgarayotgan omil (2026-05 ishlari aynan shu bosimni yaratdi). Bu **litsenziyalash bozorini** o'stiradi, lekin *dataset tayyorlash* bozorini emas.
2. **Scrape qilish mumkin bo'lgan ma'lumot yo'q bo'lganda** — ya'ni kam resursli tillar. Bu haqiqiy niche (7-bo'lim).
3. **Aniq spetsifik ma'lumot kerak bo'lganda** — ma'lum aksent, ma'lum domen, ma'lum yosh guruhi. Bu custom collection, ya'ni **operatsion biznes**, dasturiy mahsulot emas.

### 3.4 Litsenziyalash bozori shakllanmoqda (lekin boshqa shaklda)

[aipodcast.io / fiund.com](https://www.aipodcast.io/solutions/podcast-audio-licensing) — podcast back-katalogini AI labga litsenziyalaydi:
- Taklif: 48 kHz / 24-bit WAV + **word-level aligned transcripts** + verified consent + commercial training rights.
- **Har epizod uchun 100% mehmon re-clearance**, kreatorlarga shaffof revenue share (choraklik to'lov, epizod bo'yicha hisobot).
- Uch eksklyuzivlik darajasi.
- **Narx e'lon qilinmagan** — "hours, language, exclusivity tier, and metadata depth"ga bog'liq.

> **E'tibor bering:** bu kompaniya *alignment* sotmaydi — u **huquqiy tozalik + alignment**ni birga sotadi. Alignment bepul; clearance qimmat. Bu farq — butun tahlilning kalitidir.

---

## 4. Strukturaviy sabab: katta laboratoriyalarga vendor kerak emas

### 4.1 Masshtab farqi

| Loyiha | Audio hajmi | Manba | Havola |
|---|---|---|---|
| **OpenAI Whisper** | **680,000 soat** (563k ingliz + 117k boshqa 96 til) | Internetdan scrape, weak supervision | [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) |
| **Google USM** | **12,000,000 soat** belgilanmagan audio + 28 mlrd jumla matn, 300+ til, 2B parametr | Internal | [arXiv:2303.01037](https://arxiv.org/pdf/2303.01037) |
| **Meta MMS** | 1,406 til pretrain / 1,107 til ASR+TTS / 4,017 til LID | **Ommaviy diniy matnlar o'qilishi** (Yangi Ahd va shunga o'xshash) | [arXiv:2305.13516](https://arxiv.org/abs/2305.13516) |
| **Emilia** (akademik) | **101,000+ soat**, 6 til | "raw, in-the-wild speech" | [arXiv:2407.05361](https://arxiv.org/abs/2407.05361) |
| **MOSEL** (ochiq) | **950,000 soat**, YeI tillari | Ochiq manbalar agregatsiyasi | [arXiv:2410.01036](https://arxiv.org/pdf/2410.01036) |
| **Mozilla Common Voice** | 2020-07da 7,226 soat / 54 til; 2025-12 holatiga **250+ til**, CC0 litsenziya | Ko'ngilli kraudsorsing | [Wikipedia](https://en.wikipedia.org/wiki/Common_Voice) |
| **❗ Eng katta tijorat vendor paketi** | **3,125 soat** / €500,000 | Defined.ai Elite | [Defined.ai](https://defined.ai/press-room/defined-ai-launches-data-access-plan-dap-allowing-access-to-speech-datasets-at-affordable-prices) |

**Hisob (mening):**
- Vendor eng katta paketi = Whisper korpusining **0.46%**i (3,125 / 680,000).
- Whisper korpusini vendor narxida sotib olish: 680,000 × €160 = **€108.8 million**.
- USM korpusini: 12,000,000 × €160 = **€1.92 milliard**.

Hech bir lab buni qilmaydi. Ular scrape qiladi.

### 4.2 Meta MMS misoli — eng ochiq dalil

Meta 1,100+ til uchun ASR va TTS modelini qurish uchun vendor yollamadi. U **diniy matnlarning ommaviy o'qilishini** topdi — chunki Yangi Ahd yuzlab tillarda, bir ma'ruzachi, studio sifatida, matn bilan parallel yozilgan. Ya'ni **mavjud, bepul, tekstga tayanган audio korpus**. Bu — *"uzun matn + uzun audio"* g'oyasining aynan o'zi, lekin Meta buni o'z-o'zi uchun qildi va natijani bepul tarqatdi.

### 4.3 🔴 Eng og'ir dalil: mahsulot allaqachon bepul mavjud

**Emilia-Pipe** — mualliflarning o'z ta'rifi bilan *"the first open-source preprocessing pipeline designed to efficiently transform raw, in-the-wild speech data into high-quality training data with speech annotations"* ([arXiv:2407.05361](https://arxiv.org/abs/2407.05361), SLT 2024'da qabul qilingan, Amphion loyihasi).

Bu bilan 101,000+ soat, 6 tilli Emilia dataseti yaratildi va HuggingFace'da chiqarildi.

> Agar siz "avtomatik TTS dataset tayyorlash xizmati" qursangiz, raqobatchi — bepul akademik pipeline. Bu klassik **"no moat"** holati: siz bergan qiymat (segmentatsiya, alignment, filtrlash, annotatsiya) nashr qilingan, tekshirilgan va litsenziyasiz.

---

## 5. Sifat shifti — "oxirgi metr" avtomatlashtirilmaydi

### 5.1 Resursi boy tillar: review arzon

- 95% aniqlikdagi transkript 1 soat audio uchun **~5 daqiqa** odam ko'rigi talab qiladi; 85% ga tushsa — **15–20 daqiqa** ([AssemblyAI](https://www.assemblyai.com/blog/speech-recognition-cost)).
- Ya'ni ingliz tilida muammo kichik — va aynan shuning uchun ingliz tili uchun bu xizmatga ehtiyoj yo'q.

### 5.2 Kam resursli tillar: review halokatli qimmat

**🔑 Eng qimmatli raqam** — Diarra, Coulibaly, Leventhal, *"Cost Analysis of Human-corrected Transcription for Predominately Oral Languages"* ([arXiv:2510.12781](https://arxiv.org/abs/2510.12781), 2025-10-14):

> *"Through a one-month field study involving ten transcribers with native proficiency, we analyze the correction of ASR-generated transcriptions of 53 hours of Bambara voice data. We report that it takes, on average, **30 hours of human labor to accurately transcribe one hour of speech data under laboratory conditions and 36 hours under field conditions**."*

- Til: Bambara (Mali, Manding guruhi) — "predominantly oral", past savodxonlik.
- **Bu ASR chiqishini TUZATISH vaqti**, noldan yozish emas.
- 10 ona tili sohibi, 53 soat audio, 1 oy.

**Taqqoslash (mening hisobim):**
```
Ingliz, 95% ASR:     ~0.08 odam-soat / audio-soat
Bambara, ASR+tuzatish: 30–36 odam-soat / audio-soat
                       → ~375–450× farq
```

**Narxga aylantirganda (illyustrativ hisob, vendor bayonoti emas):**
- Agar transkriberga soatiga $3 to'lansa: 30 × $3 = **$90 / audio-soat**
- Soatiga $10: **$300 / audio-soat**
- Soatiga $20: **$600 / audio-soat**

Bu raqamlar vendor katalogi narxlari ($60–150/soat kam resursli tillar uchun) bilan **bir xil darajada yoki undan yuqori**. Ya'ni: **kam resursli tilda sifatli dataset yaratishning haqiqiy mehnat narxi vendorlar so'rayotgan narxdan yuqori.** Bu marja yo'qligini bildiradi.

### 5.3 Nega avtomatlashtirish bu muammoni hal qilmaydi

- ASR aniqligi past bo'lgan tilda tuzatish vaqti chiziqli emas, **portlab o'sadi** — WER qancha yuqori bo'lsa, odam shuncha ko'p qayta tinglashi kerak.
- TTS dataseti uchun yana qo'shimcha talablar bor: punktuatsiya, katta-kichik harf, raqam normalizatsiyasi, prosodiya, orfografik izchillik. O'zbek tilida bunga **alifbo masalasi** (kirill/lotin) qo'shiladi — FeruzaSpeech "ikkala transkripsiyani bergan birinchi dataset" deb ta'kidlangani bejiz emas ([arXiv:2410.00035](https://arxiv.org/pdf/2410.00035)).
- Ya'ni avtomatlashtirish WER'ni 50%dan 30%ga tushirsa ham, odam ko'rigi hali ham oylik mehnat bo'lib qoladi. **Bu — mahsulot emas, xizmat ko'rsatish biznesi.**

---

## 6. Kimlar urinib ko'rdi va nima bo'ldi?

### 6.1 cielo24 — eng aniq failure case

**Hikoya** ([SimpleClosure case study](https://simpleclosure.com/customer-stories/cielo24/)):

| Bosqich | Nima bo'ldi |
|---|---|
| **2012** | Santa Barbara startupidan spin-out. Maqsad: "making spoken media searchable" — onlayn platformalar va universitetlar uchun audio/video transkripsiya |
| **Evolyutsiya** | Xom transkripsiyani **strukturalashgan, time-aligned datasetlarga** aylantiradigan vositalar qurdi |
| **2014–2015** | AI mainstream bo'lishidan ancha oldin ML bilan tajriba: o'sib borayotgan **labeled speech corpus**ini modellarni o'qitish uchun ishlatish yo'llarini izladi |
| **O'sish** | VC puli; NY va Kaliforniyada ofislar; **~70 xodim** |
| **2019** | Shanna Johnson (avval COO) CEO bo'lib qaytdi; **18 oyda foydalilikni tikladi** |
| **Yopilish** | Eng katta mijozni yo'qotdi + bozor o'zgardi → davom etish biznes ma'noga ega bo'lmadi |
| **Aktiv sotuvi** | 10 yillik ma'lumot aktivlari **"mid-six figures"**ga sotildi (SimpleClosure orqali topilgan xaridor, "language technology sohasida ilg'or tashkilot") |

Johnson'ning so'zlari: *"It had never crossed my mind that our data could be part of the company's legacy."* Va: *"If you are going to fail, fail fast. But when it is time to shut down, do it right."*

> **🔴 Xulosa:** 10 yil, 70 xodim, VC puli va to'liq time-aligned speech korpusi — **$1 milliondan kam**ga sotildi. Bu — speech-dataset biznesining bozor bahosi. Agar bu biznes model ishlaganda, xaridorlar navbatda turardi.

### 6.2 Defined.ai / DefinedCrowd — pivot, yopilish emas

- 2015-yilda Dr. Daniela Braga tomonidan tashkil etilgan, Seattle + Lissabon R&D ([Crunchbase](https://www.crunchbase.com/organization/definedcrowd)).
- **2022-yilda DefinedCrowd → Defined.ai rebrand.** O'zlarining ta'rifi bilan: *"a pivotal shift — from a data provider to an end-to-end enabler of ethical, performant AI"*; "crowdsourced data gathering"dan "comprehensive AI platform and marketplace"ga o'tish ([press-relize](https://defined.ai/press-room/definedcrowd-rebrands-as-defined-ai-reflecting-expanded-position-as-a-developer-platform-for-artificial-intelligence), [bold-awards tahlili](https://bold-awards.com/speech-datasets-provider-pivots-for-new-era-of-ai-development/)).
- **Bugun (2026-10) hali faol** — ISO 42001 sertifikati, 1.6M+ ekspert, 150+ davlat, 500+ til/dialekt ([defined.ai](https://defined.ai/)).
- Lekin endi ular **nima sotadi:** tayyor datasetlar + custom collection + **data evaluation + model evaluation** + machine translation + conversational AI yechimlari. Ya'ni dataset sotuvi — portfelning bir qismi, mahsulot emas.
- **Yopilish, insolvensiya yoki mass layoff haqida dalil topilmadi — ma'lumot topilmadi.** (Men buni maxsus qidirdim.)

> **Xulosa:** bu sohada eng yirik, eng yaxshi mablag' bilan ta'minlangan "speech dataset kompaniyasi" ham **sof dataset sotuvidan platformaga va evaluation xizmatlariga o'tdi**. Bu signal.

### 6.3 Appen — annotatsiya biznesining qiyinchiligi

- 2023 yil daromadi **US$273.0 million**, pasayish ko'rsatkichi bilan; xodimlar 2017-da 1,000+ FT dan 2024-da jami 1,000 ga tushgan ([Wikipedia: Appen (company)](https://en.wikipedia.org/wiki/Appen_(company))).
- **Google shartnomasining bekor qilinishi, aksiya narxining tushishi va layoff raqamlari — men tekshirgan manbalarda tasdiqlanmadi (ma'lumot topilmadi).** Bu haqda ko'p gapiriladi, lekin birlamchi manbani olishga muvaffaq bo'lmadim; mustaqil tekshirish kerak.
- Appen narxlash modeli: qayta ishlangan qator uchun, **transkripsiya qilingan audio-soat uchun**, yoki faol foydalanuvchi uchun — aniq stavkalar e'lon qilinmagan ([growthsharematrix](https://growthsharematrix.com/products/appen-marketing-mix)).

### 6.4 15.ai — ochiq loyihaning o'limi

**Nega bu muhim:** 15.ai — tarixda eng mashhur "kichik dataset bilan TTS" loyihasi, va uning hikoyasi sizning g'oyangizning to'g'ri va noto'g'ri tomonlarini birga ko'rsatadi ([Wikipedia: 15.ai](https://en.wikipedia.org/wiki/15.ai)):

| Sana | Voqea |
|---|---|
| **2020-03-02** | Ishga tushdi. Bepul, ro'yxatdan o'tish shart emas, nokommersial |
| — | **Dataset:** "Pony Preservation Project" — /mlp/ 4chan bordidagi ko'ngillilar **qo'lda trim, denoise, transkripsiya va emotion-tag qilgan** minglab ovoz qatorlari |
| 2019 | Muallif "avval kerak bo'lgandan **75% kam** o'qitish ma'lumoti bilan" ilg'or modellarni takrorlashni ko'rsatdi. Nom "15" — 15 soniya audiodan ovoz klonlash imkoniga ishora |
| **2022-01-14** | **Voiceverse NFT skandali:** Voiceverse kompaniyasi 15.ai bilan audio generatsiya qilib, uni o'z texnologiyasi sifatida ko'rsatdi va NFT qilib sotdi. Voiceverse: marketing jamoasi "demo tezda yaratish uchun to'g'ri atribut bermasdan" ishlatgan |
| **2022-09** | **Oflayn** — "AI va mualliflik huquqi atrofidagi huquqiy muammolar" tufayli |
| 2025-05-18 | 15.dev voris loyihasi ishga tushdi |
| **2026-05-28** | 15.dev **noma'lum muddatga yopilishi** e'lon qilindi; muallif Artist Alley marketpleysiga o'tdi |

> **🔴 Ikki xulosa:**
> 1. **Dataset tayyorlash eng qiyin qismi edi — va u pul bilan emas, ko'ngilli mehnat bilan hal qilindi.** Minglab ovoz qatorini qo'lda tozalash, transkripsiya va emotion-tag qilish — 4chan bordi bepul qildi. Buni xizmat sifatida sotib bo'lmaydi, chunki muhabbat bilan qilinganda bepul.
> 2. **Loyiha texnik sabab bilan emas, huquqiy sabab bilan o'ldi.** Bu 1-bo'limdagi barcha dalillarni tasdiqlaydi.

### 6.5 Boshqa izlar

- **Ochiq kodli pipeline'lar tark etilishi:** maxsus izlab, aniq "tark etilgan TTS dataset pipeline" ro'yxatini tasdiqlangan manba bilan topa olmadim — **ma'lumot topilmadi**. Lekin aksi aniq: Emilia-Pipe, NeMo, Common Voice faol va bepul — ya'ni muammo tark etilish emas, **kommoditizatsiya**.
- **"TTS dataset as a service" startupi:** aynan shu pozitsiyada ishlagan va yopilgan/pivot qilgan kompaniyani qidirdim — cielo24dan boshqa aniq misol topilmadi (**ma'lumot topilmadi**). Bu o'z-o'zidan ma'noli: kategoriya shunchalik kichik ki, unda hatto failure'lar ham yo'q.

---

## 7. Imkoniyat qayerda? — halol baho

### 7.1 Umumiy jadval: imkoniyat vs to'siq

| Imkoniyat yo'nalishi | Nega ishlashi mumkin (dalil) | To'siq (dalil) | Baho |
|---|---|---|---|
| **Umumiy "TTS dataset tayyorlash SaaS"** | — | Emilia-Pipe bepul va ochiq ([arXiv:2407.05361](https://arxiv.org/abs/2407.05361)); ASR $0.15–0.36/soat; labs 680k–12M soat scrape qiladi | ❌ **Qilmang.** Moat yo'q, mijoz yo'q |
| **Labs uchun katta hajmli dataset sotish** | Litsenziyalash bozori o'smoqda; podcast litsenziyalash kompaniyalari paydo bo'ldi ([aipodcast.io](https://www.aipodcast.io/solutions/podcast-audio-licensing)) | Masshtab imkonsiz: eng katta vendor paketi 3,125 soat, Whisper 680,000 soat. cielo24 10 yillik korpusi <$1M ga sotildi | ❌ **Kapital yo'q bo'lsa — yo'q** |
| **Kam resursli tillar uchun dataset ishlab chiqarish** | Gates Foundation → Masakhane **$4,998,785** (2025-09, 40 oy) ([Gates](https://www.gatesfoundation.org/about/committed-grants/2025/09/inv-088221)); Masakhane AI hub **CAD 5,360,000** (IDRC, 40 Afrika tili) ([IDRC](https://idrc-crdi.ca/en/what-we-do/projects-we-support/project/masakhane-ai-languages-hub)); RESPIN 10,000+ soat 9 hind tili (Gates) ([OpenReview](https://openreview.net/pdf/a75242153008713de8b43aa6997257a3b1b0e85f.pdf)); Lacuna Fund — 24 til dataseti, KenSpeech/Ghana/Ethio ([Lacuna Fund](https://lacunafund.org/datasets/)) | Bambara: **30–36 odam-soat / audio-soat** ([arXiv:2510.12781](https://arxiv.org/abs/2510.12781)). Grant pullari **ochiq** dataset talab qiladi — qayta sotish mumkin emas | ⚠️ **Ha, lekin grant/xizmat biznesi** |
| **On-prem / self-hosted tooling** | Maxfiylik audio tashqariga chiqishini taqiqlaydi: GDPR Art. 9 explicit consent + DPIA; BIPA; sog'liq/call-center audio | Mijozlar soni kichik; har biri bespoke integratsiya talab qiladi; narx bosimi (ochiq kodga nisbatan) | ⚠️ **Eng realistik yo'l** |
| **"Bring your own recordings" (BYOR)** | Mijoz huquqni o'zi olib keladi → siz biometrik/copyright zanjiridan chiqasiz. SAG-AFTRA/ELVIS/AB 2602 modeli aynan shu: rozilik ijrochidan keladi | Mijoz o'zi uchun bir marta ishlatadi → recurring revenue past; bozor hajmi kichik | ✅ **Huquqiy jihatdan eng toza** |
| **Huquqiy tozalash + alignment birgalikda** | aipodcast.io modeli: 48kHz/24-bit WAV + **word-level aligned transcripts** + verified consent + commercial rights + har epizod uchun 100% re-clearance + revenue share | Bu kontent biznesi, dasturiy mahsulot emas — kontent egalaridan tarmoq talab qiladi | ✅ **Qiymat shu yerda, lekin boshqa biznes** |

### 7.2 O'zbek tili — aniq raqamlar

| Dataset | Hajm | Tavsif | Havola |
|---|---|---|---|
| **USC** (Uzbek Speech Corpus) | **105 soat** | 958 ma'ruzachi, turli hudud va yosh; ASR uchun, TTS/translation uchun ham ishlatish mumkin | [arXiv:2107.14419](https://arxiv.org/pdf/2107.14419), [ISSAI](https://issai.nu.edu.kz/uzbek-asr/) |
| **Common Voice (uz)** | **266 soat** | 2000+ ma'ruzachi, CC0 | Qidiruv natijasi (Common Voice sahifasini to'g'ridan-to'g'ri olib bo'lmadi) |
| **FeruzaSpeech** | **60 soat** | Toshkentlik bitta ayol ona tili sohibi; **kirill va lotin transkripsiyani birga bergan birinchi dataset** | [arXiv:2410.00035](https://arxiv.org/pdf/2410.00035) |

**Jami ochiq o'zbek speech data ≈ 430 soat.** Taqqoslang: Whisper'ning ingliz qismi 563,000 soat. Nisbat ≈ **1 : 1,300**.

Bu haqiqiy bo'shliq. Lekin diqqat: bo'shliq **ehtiyoj**ni bildiradi, **to'lovga qodir xaridor**ni bildirmaydi. 430 soatni 4,300 soatga chiqarish uchun kim to'laydi?

Dalillarga ko'ra javob: **xalqaro rivojlanish fondlari va hukumatlar** (Gates, IDRC, Lacuna Fund), **universitetlar**, va ehtimol davlat raqamlashtirish programmalari. Bularning hammasi **grant/tender** mantig'ida ishlaydi, SaaS obunasi mantig'ida emas. Va ularning hammasi natija **ochiq** bo'lishini talab qiladi.

### 7.3 Haqiqiy tavsiya — agar qursangiz, qanday shaklda

**❌ Qilmang:**
- "Audio yuklang, biz dataset qaytaramiz" SaaS — Emilia-Pipe bepul.
- Scrape qilingan audiokitob/podcastdan dataset yig'ib sotish — 2026-05 BIPA ishlari aynan shu.
- "100 tilda tayyor TTS dataset marketpleysi" — Defined.ai bu yo'ldan qaytdi.

**⚠️ Ehtiyot bilan:**
- Kam resursli til datasetini **grant pulida** ishlab chiqarish. Realistik, lekin bu **mehnat biznesi** (30–36 odam-soat/audio-soat) va natija ochiq bo'ladi. Marja — xizmat marjasi, mahsulot marjasi emas.

**✅ Eng kuchli pozitsiyalar:**

1. **Self-hosted, on-prem dataset-prep toolkit** — audio hech qachon mijoz infratuzilmasidan chiqmaydi. Sotish argumenti huquqiy: GDPR Art. 9 DPIA, BIPA, sektor regulyatsiyasi. Mijozlar: call-center, sog'liq, davlat, media arxivlar. Narx — litsenziya + support, dataset emas.
2. **"Bring your own recordings" workflow** — mijoz ovoz egasi bilan roziligi (SAG-AFTRA/AB 2602 uslubidagi "reasonably specific" rozilik) va yozuvlarni olib keladi; siz faqat alignment, segmentatsiya, normalizatsiya, sifat ko'rigini avtomatlashtirasiz. Siz biometrik ma'lumot zanjirida **processor** bo'lasiz, controller emas — bu riskni keskin kamaytiradi.
3. **Consent va provenance infratuzilmasi** — sizning haqiqiy differentiator'ingiz pipeline emas, **audit izi** bo'lishi mumkin: har segment uchun kim, qachon, qaysi rozilik bilan, qaysi yurisdiksiyada. EU AI Act Art. 50 (machine-readable marking, 2026-08-02dan) va BIPA/ELVIS compliance buni *majburiy* qilmoqda. aipodcast.io shu yo'nalishda harakat qilmoqda, lekin bozor to'lmagan.
4. **O'zbek/turkiy tillar uchun vertikal** — til bilimi sizning moat'ingiz. Lotin/kirill ikki tomonlama transkripsiya, o'zbek orfografiyasi, raqam va qisqartma normalizatsiyasi — bu Emilia-Pipe'da yo'q va g'arbiy jamoalar qila olmaydi. Buni **mahsulot ichidagi til moduli** sifatida qurish — pipeline'ning o'zini sotishdan ko'ra mustahkam.

---

## 8. Eng muhim xulosa — bir jumlada

**Bu mahsulot kategoriyasi emas, chunki qiymat zanjirining avtomatlashtiriladigan qismi bepul va ochiq (Emilia-Pipe, ASR $0.15/soat), avtomatlashtirilmaydigan qismi esa odam mehnati (Bambara: 30–36 odam-soat/audio-soat), eng yirik xaridorlar o'zlari 680,000–12,000,000 soat scrape qiladi, va o'rtada qolgan har qanday vositachi ovoz biometrikasi (BIPA $1,000–5,000/buzilish), raqamli nusxa qonunlari (ELVIS Act — hatto tool sotuvchiga ham javobgarlik) va kasaba uyushmasi rozilik rejimlari (SAG-AFTRA IMA) orqali javobgarlikni o'ziga oladi.**

Imkoniyat **pipeline sotishda emas** — u **rozilik, provenance va til ekspertizasini** sotishda.

---

## 9. Tekshirilmagan / topilmagan ma'lumotlar (halol ro'yxat)

Quyidagilarni tasdiqlab bo'lmadi — ulardan foydalanishdan oldin mustaqil tekshirish kerak:

1. **BIPA 740 ILCS 14/10 "biometric identifier" ta'rifida "voiceprint" so'zi bor-yo'qligi** — qonun matnini ilga.gov va justia.com'dan olishga uch marta urindim, ikkalasi ham bloklandi/404. Ikkilamchi manbalar (Whole Foods kelishuvi, Zaluda v. Apple, 2026-05 ishlari) voiceprint BIPA ostida da'vo qilinishini **amalda** tasdiqlaydi, lekin statut matnini keltira olmayman.
2. **BIPA 2024-yilgi o'zgartirishi (SB 2979 — per-scan zarar to'planishini cheklash)** — manbalarda topilmadi. Agar BIPA risk hisobini qilsangiz, bu muhim.
3. **California AB 2602 va AB 1836 ning aniq kuchga kirish sanasi** — faqat "2024-yil sentabrda imzolangan" tasdiqlandi.
4. **Tennessee ELVIS Act kuchga kirish sanasi** — manbalar qarama-qarshi (2024-07-01 vs 2024-07-21).
5. **SAG-AFTRA 2025 IMA dagi aniq digital replica minimal stavkalari** — rasmiy PDF (sagaftra.org) 403 qaytardi.
6. **Appen'ning Google shartnomasini yo'qotishi, aksiya tushishi, layoff raqamlari** — tasdiqlanmadi.
7. **John Hancock voiceprint hukmining aniq matni va sanasi** — maqola fetch qilinmadi.
8. **Audiokitob narrator stavkalari va AI narration narxi** — Publishers Weekly va Marketplace maqolalarida raqam yo'q / fetch bloklandi.
9. **aipodcast.io / fiund.com soat narxlari** — e'lon qilinmagan.
10. **Defined.ai'ning chegirmasiz "standart" soat narxi** — e'lon qilinmagan (€227–356/soat — mening hisobim, ularning bayonoti emas).
11. **Common Voice'ning jami soati va o'zbek tili aniq soati** — commonvoice.mozilla.org bloklandi; o'zbek 266 soat raqami qidiruv natijasidan, birlamchi manbadan emas.
12. **Lehrman v. Lovo docket raqami** — manbada ko'rsatilmagan (reporter sitatasi: 790 F.Supp.3d 348 (S.D.N.Y. 2025)).
13. **EDPB'ning voiceprint bo'yicha birlamchi ko'rsatmasi** — faqat ikkilamchi tahlillar orqali.
14. **Scarlett Johansson OpenAI'ga da'vo arizasi berdimi** — dalil topilmadi (faqat advokat xatlari).

---

## 10. Manbalar

### Huquq — BIPA va ovoz biometrikasi
- https://www.hunton.com/privacy-and-cybersecurity-law-blog/whole-foods-settles-bipa-voiceprint-class-action0
- https://www.cbsnews.com/chicago/news/illinois-class-action-lawsuits-voices-stolen-train-ai-bipa/
- https://www.sgtlaw.com/cases/apple-siri-bipa-litigation
- https://www.findlaw.com/legalblogs/third-circuit/is-your-voiceprint-protected-amazons-biometric-privacy-battle-in-illinois/
- https://topclassactions.com/lawsuit-settlements/lawsuit-news/john-hancock-voiceprint-ruling-highlights-limits-of-illinois-biometric-privacy-law/
- https://en.wikipedia.org/wiki/Biometric_Information_Privacy_Act
- https://www.segalmccambridge.com/blog/technology-cyber-risk-client-alert-evolution-of-bipa-voiceprint-claims-in-illinois-persists/ (404)

### Huquq — AI o'qitish uchun ovoz scrape qilish da'volari
- https://www.thebookseller.com/news/us-audiobook-narrators-sue-google-for-using-their-voices-in-ai-training
- https://www.ciplawyer.com/articles/159009.html
- https://www.playwire.com/blog/googles-voice-ai-lawsuit-is-a-warning-for-every-content-creator
- https://boxcommons.org/blog/audio-data-liability-frontier/

### Huquq — Lehrman v. Lovo
- https://www.loeb.com/en/insights/publications/2025/07/lehrman-v-lovo-inc
- https://www.crowell.com/en/insights/client-alerts/fundamental-copyright-principles-underscored-in-ai-context-voice-attributes-are-not-protectable
- https://case-law.vlex.com/vid/lehrman-v-lovo-inc-1098786772
- https://www.cbsnews.com/news/two-voice-actors-sue-ai-company-lovo/
- https://www.offitkurman.com/offit-kurman-blogs/ai-voice-cloning-lawsuit-voice-actor-rights

### Huquq — GDPR
- https://www.linkedin.com/pulse/when-voice-special-category-personal-data-under-gdpr-janvier-parewyck
- https://www.dilr.ai/blog/ai-voice-biometric-data-security-enterprise
- https://ypai.ai/blog/compliance/gdpr-compliant-speech-data-collection-europe/

### Huquq — EU AI Act
- https://artificialintelligenceact.eu/article/50/
- https://artificialintelligenceact.eu/transparency-rules-article-50/
- https://truescreen.io/insights/ai-act-article-50-labelling-synthetic-content-august-2026/
- https://hard2bit.com/en/blog/ai-act-article-50-ai-transparency-chatbots-deepfakes/
- https://www.jonesday.com/en/insights/2026/01/european-commission-publishes-draft-code-of-practice-on-ai-labelling-and-transparency
- https://digital-strategy.ec.europa.eu/en/factpages/quick-facts-transparency-rules-ai-systems

### Huquq — digital replica (AQSh shtatlari)
- https://www.identity.org/what-is-tennessees-elvis-act/
- https://www.spirelight.ai/guides/voice-cloning-laws
- https://www.insideglobaltech.com/2024/09/11/california-passes-digital-replica-legislation-as-congress-considers-federal-approach/
- https://www.globalpolicywatch.com/2024/09/california-passes-digital-replica-legislation-as-congress-considers-federal-approach/
- https://www.fenwick.com/insights/publications/californias-new-ai-laws-limit-uses-of-digital-likeness
- https://www.proskauer.com/blog/california-enacts-generative-ai-law-addressing-digital-replicas-of-performers
- https://www.manatt.com/insights/newsletters/client-alert/california-enacts-a-host-of-new-ai-and-digital-rep

### SAG-AFTRA
- https://www.dglaw.com/sag-aftras-new-video-game-agreement/
- https://www.aitrace.org/company/sag-aftra/practice/1f131e05-3c62-495a-9596-4aa1930e8db2
- https://deadline.com/2025/07/sag-aftra-video-game-agreement-approved-strike-1236453741/
- https://en.wikipedia.org/wiki/2024%E2%80%932025_SAG-AFTRA_video_game_strike
- https://www.sagaftra.org/sag-aftra-and-replica-studios-introduce-groundbreaking-ai-voice-agreement-ces
- https://www.sagaftra.org/sites/default/files/2025-06/2025%20Interactive%20Media%20(Video%20Game)%20Agreement%20Summary.pdf (403)
- https://www.voiceovernews.com/sag-aftra-ends-strike-ai-voice-deal/

### Scarlett Johansson / OpenAI
- https://www.cnbc.com/2024/05/20/scarlett-johansson-says-openai-ripped-off-her-voice-.html
- https://techcrunch.com/2024/05/20/openai-to-remove-chatgpts-scarlett-johansson-like-voice
- https://www.today.com/popculture/news/scarlett-johansson-openai-voice-sky-rcna153188
- https://fortune.com/2024/05/20/scarlett-johansson-chatgpt-sky-voice-lawyer-sam-altman-openai-her

### Manba materiali / mualliflik huquqi
- https://librivox.org/pages/public-domain/
- https://www.gutenberg.org/policy/permission.html
- https://legalblogs.wolterskluwer.com/copyright-blog/the-new-copyright-directive-text-and-data-mining-articles-3-and-4/
- https://creativecommons.org/wp-content/uploads/2021/12/CC-Statement-on-the-TDM-Exception-Art-4-DSM-Final.pdf
- https://onlinelibrary.wiley.com/doi/full/10.1111/jwip.12285
- https://www.dlapiper.com/en-us/insights/blogs/mse-today/2025/training-ai-models-uk-government-proposes-eu-style-opt-out-copyright-exception
- https://legalblogs.wolterskluwer.com/copyright-blog/the-uk-government-moves-forward-with-a-text-and-data-mining-exception-for-all-purposes/
- https://academic.oup.com/grurint/article/71/8/685/6650009

### Iqtisod / narxlar
- https://defined.ai/press-room/defined-ai-launches-data-access-plan-dap-allowing-access-to-speech-datasets-at-affordable-prices
- https://www.speechdata.ai/guides/how-to-buy-speech-data
- https://www.assemblyai.com/pricing
- https://www.assemblyai.com/blog/speech-recognition-cost
- https://diyai.io/ai-tools/speech-to-text/openai-whisper-api-pricing-2026/
- https://www.notta.ai/en/blog/transcription-rate
- https://sonix.ai/how-much-does-transcription-cost
- https://growthsharematrix.com/products/appen-marketing-mix
- https://www.aipodcast.io/solutions/podcast-audio-licensing
- https://fiund.com/
- https://www.publishersweekly.com/pw/by-topic/digital/content-and-e-books/article/97387-the-audiobook-business-learns-to-embrace-ai.html
- https://www.marketplace.org/story/2026/09/30/will-ai-take-over-the-audiobook-industry (403)

### Masshtab / modellar
- https://arxiv.org/abs/2212.04356 (Whisper, 680k soat)
- https://arxiv.org/pdf/2303.01037 (Google USM, 12M soat)
- https://arxiv.org/abs/2305.13516 (Meta MMS, diniy matnlar)
- https://arxiv.org/abs/2407.05361 (Emilia + Emilia-Pipe, 101k soat, ochiq pipeline)
- https://arxiv.org/pdf/2410.01036 (MOSEL, 950k soat)
- https://en.wikipedia.org/wiki/Common_Voice

### Oxirgi metr / annotatsiya narxi
- https://arxiv.org/abs/2510.12781 (Bambara: 30–36 odam-soat/audio-soat)

### Failure / pivot
- https://simpleclosure.com/customer-stories/cielo24/
- https://strategicsignal.ai/ai-data-transactions/deals/ss-aidt-2026-0001/
- https://defined.ai/press-room/definedcrowd-rebrands-as-defined-ai-reflecting-expanded-position-as-a-developer-platform-for-artificial-intelligence
- https://bold-awards.com/speech-datasets-provider-pivots-for-new-era-of-ai-development/
- https://defined.ai/
- https://www.crunchbase.com/organization/definedcrowd
- https://en.wikipedia.org/wiki/Appen_(company)
- https://en.wikipedia.org/wiki/15.ai

### Kam resursli tillar — talab va mablag'
- https://www.gatesfoundation.org/about/committed-grants/2025/09/inv-088221
- https://idrc-crdi.ca/en/what-we-do/projects-we-support/project/masakhane-ai-languages-hub
- https://lacunafund.org/datasets/
- https://openreview.net/pdf/a75242153008713de8b43aa6997257a3b1b0e85f.pdf (RESPIN-S1.0)
- https://theconversation.com/african-languages-for-ai-the-project-thats-gathering-a-huge-new-dataset-266371
- https://arxiv.org/pdf/2510.05644 (African Languages Lab)

### O'zbek / Markaziy Osiyo
- https://arxiv.org/pdf/2107.14419 (USC, 105 soat)
- https://issai.nu.edu.kz/uzbek-asr/
- https://arxiv.org/pdf/2410.00035 (FeruzaSpeech, 60 soat)
- https://www.mdpi.com/2076-3417/15/24/12880 (turkiy tillar ko'p tilli korpus)

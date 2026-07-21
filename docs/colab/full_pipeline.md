# ScentAI Stage 3 - Colab Calistirma Rehberi

Stage 3, calisan inference ve retrieval servislerini tek tam pipeline icinde
birbirine baglar. Bu notebook sonunda serbest kullanici sorgulari dogrudan
ScentAI cevabina donusur.

## 1. Colab'de acilacak dosya

Bilgisayarindan su notebook'u Google Colab'de ac:

`notebooks/full_pipeline_colab.ipynb`

Notebook self-contained durumdadir. Su dosyalari ayrica Colab'e yukleme:

- `orchestrator.py`
- `retrieval_service.py`
- Stage 1 veya Stage 2 notebook'lari

Bu kaynaklar Stage 3 notebook'unun icine gomuludur.

## 2. Drive'da bulunmasi gereken yapilar

```text
MyDrive/
└── Perfume-Dataset/
    ├── chroma_db_bge_m3/
    │   ├── chroma.sqlite3
    │   └── diger Chroma dosyalari
    ├── scentai_catalog.sqlite3
    └── models/
        ├── scentai-gemma-4-12b-it-full-fastmodel-lora/
        │   └── best_lora_adapter/
        │       ├── adapter_config.json
        │       └── adapter_model.safetensors
        └── scentai-gemma-4-12b-it-pilot-fastmodel-lora/
            └── best_lora_adapter/
                ├── adapter_config.json
                └── adapter_model.safetensors
```

`chroma_db_bge_m3` tam klasor olarak bulunmalidir. Icerisinden yalnizca
`chroma.sqlite3` dosyasini almak yeterli degildir.

Notebook once vLLM-uyumlu full-run adapter'ini kullanir. Full adapter yoksa veya
DoRA/rank/target-module yapisi bu vLLM yoluyla uyusmuyorsa bunu ekrana yazar ve
pilot adapter'a duser.

## 3. Colab runtime

Colab menusunden:

```text
Runtime > Change runtime type
Hardware accelerator: A100 GPU
Runtime shape: High-RAM
```

Yeni ve temiz bir Colab oturumu kullan. Ayni oturumda once eski inference,
retrieval, Unsloth veya vLLM notebook'larini calistirma.

## 4. Opsiyonel Hugging Face token

Colab Secrets bolumunde `HF_TOKEN` adli bir secret kullanabilirsin. Model
repository'si public olarak erisilebildigi surece token zorunlu degildir.

## 5. Calistirma

Notebook'u actiktan sonra:

```text
Runtime > Run all
```

Huceleri atlama veya siralarini degistirme. Notebook su islemleri yapar:

1. `uv` launcher kurar.
2. vLLM/CUDA icin ayri Python ortami olusturur.
3. Chroma/BGE-M3 icin CPU-only ayri Python ortami olusturur.
4. Iki dependency grafigini ayri ayri test eder.
5. Drive varliklarini dogrular.
6. Chroma ve SQLite verilerini `/content` altina kopyalar.
7. Retrieval servisini `127.0.0.1:8020` adresinde baslatir.
8. Gemma 4 + ScentAI LoRA servisini `127.0.0.1:8010` adresinde baslatir.
9. Base model ve LoRA endpoint'lerini test eder.
10. Planner, retrieval, base-Gemma danisman cevabi ve validator pipeline'ini kurar; base cevap gecmezse LoRA'yi tek onarim denemesi olarak kullanir.
11. Turkce/Ingilizce dil ve performans kalibrasyonu dahil end-to-end kalite senaryolarini calistirir.
12. Genel isteklerde semantik uygunluk ile katalog popülerligi dengesini test eder.
13. Ardisik sorgularda eski onerileri tekrar etmeyen sohbet hafizasini test eder.
14. Legacy LoRA, advisor LoRA ve advisor base Gemma cevaplarini ayni adaylarla A/B test eder.
15. Vanilla, spicy, oud, rose, musk, amber ve diger trait'lerde isim bias audit'i calistirir.
16. Dokuz kategorideki 120 sabit sorguyla final evaluation kosusunu yapar; her cevabi aninda Drive'a kaydeder.
17. Otomatik kalite kapilarini, ozet raporu ve 40 satirlik insan inceleme CSV'sini uretir.

Kimlik cozumleyici marka adlarindan benzersiz bas harf kisaltmalarini katalog
acilisinda otomatik uretir. Ayrica yaygin ama cakisan `YSL`, `LV`, `JPG`, `MFK`,
`PDM`, `CDG`, `TF`, `CH` ve `D&G` kisaltmalari kucuk bir dogrulanmis alias
tablosuyla desteklenir. `EDP`, `EDT`, `EDC`, yazim hatali `EPD/ETD` ve parfum
surum adlari ayri kimlik sinyalleri olarak puanlanir; belirtilen surum daha
populer bir flankere feda edilmez. Marka kesin cozuldugunde urun adindaki makul
yazim hatalari yalnizca o markanin katalog satirlari icinde aranir.

Ilk calistirmada vLLM kurulumu, Gemma 4 agirliklarinin indirilmesi, BGE-M3
yuklenmesi ve Chroma kopyasi nedeniyle uzun bekleme normaldir.

## 6. Basari kriteri

Asagidaki satiri gorursen Stage 3 basarilidir:

```text
STAGE 3 END-TO-END CONTRACT TEST: PASSED
```

Rapor su konuma kaydedilir:

```text
MyDrive/Perfume-Dataset/runs/stage3_pipeline_report.json
```

Retrieval isim-bias raporu da su konuma kaydedilir:

```text
MyDrive/Perfume-Dataset/runs/stage3_retrieval_bias_audit.json
```

Populerlik ve sohbet hafizasi sozlesmesi su konuma kaydedilir:

```text
MyDrive/Perfume-Dataset/runs/stage3_popularity_conversation_contract.json
```

Danisman promptu A/B raporu su konuma kaydedilir:

```text
MyDrive/Perfume-Dataset/runs/stage3_advisor_ab_report.json
```

Final evaluation dosyalari su klasore kaydedilir:

```text
MyDrive/Perfume-Dataset/runs/final_evaluation/
```

120 vakalik kosu kesilirse ayni hucre yeniden calistirildiginda tamamlanan vaka
kimlikleri atlanir. Model, adapter, katalog veya orchestrator degistiyse eski
klasoru arsivleyip temiz bir final kosusu baslat.

A/B testi uc sorgu x uc varyant olmak uzere dokuz cevap uretir. Bu nedenle ilk
Run All normal Stage 3 testinden birkac dakika daha uzun surebilir. Bu tek
seferlik tani testi bittikten sonra kazanan cevap yolunu sabitleyip legacy/base
karsilastirma cagirilari kaldirilabilir.

## 7. Kendi sorgunu calistirma

En alttaki hucrede yalnizca `my_query` metnini degistir:

```python
my_query = "I want a versatile woody fragrance for autumn evenings, but no oud. Recommend 3 options."
my_result = ask_scentai(my_query)
```

Ayni oturumda dogal bir takip mesaji gonderebilirsin. Onceki sartlar korunur ve
daha once gercekten onerilen parfumler yeni adaylardan cikarilir:

```python
my_result = ask_scentai("Baska uc secenek istiyorum.")
```

Alakasiz yeni bir konuya gececeksen hafizayi temizle:

```python
reset_scentai_session()
```

Pipeline genel amacli bir sohbet botu degildir; parfum danismanligi icinde
cok turlu, veritabanina dayali sohbet tasarlanmistir.

Sonuclar otomatik olarak su dosyaya eklenir:

```text
MyDrive/Perfume-Dataset/runs/stage3_interactive_results.jsonl
```

## 8. Bu asamada yapilmayacaklar

- Notebook kernel'ine elle `torch`, `vllm`, `chromadb` veya `transformers` kurma.
- Pinned dependency surumlerini degistirme.
- Stage 1 ve Stage 2 servislerini baska notebook'lardan ayni anda baslatma.
- Model yuklenirken hucreyi tekrar tekrar calistirma.
- Chroma klasorunu ZIP olarak birakma; Drive'da acilmis klasor olmali.

Bu Stage 3 notebook'u tam yerel Colab pipeline'idir. Public web API veya arayuz
deployment'i sonraki ayri asamadir.

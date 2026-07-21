# ScentAI Stage 2 - Colab Calistirma Rehberi

Bu asamada yalnizca retrieval sistemini test ediyoruz. Gemma, LoRA, vLLM veya
Unsloth calistirilmiyor. GPU gerekli degil.

## 1. Colab'e yuklenecek dosya

Bilgisayarindan asagidaki notebook'u Google Colab'de ac:

`notebooks/retrieval_colab.ipynb`

Notebook self-contained durumdadir. `retrieval_service.py` notebook'un icine
gomuludur; bu Python dosyasini Drive'a veya Colab'e ayrica yukleme.

## 2. Google Drive'da bulunmasi gereken veriler

Notebook calismadan once Drive yapisi tam olarak soyle olmali:

```text
MyDrive/
└── Perfume-Dataset/
    ├── chroma_db_bge_m3/
    │   ├── chroma.sqlite3
    │   └── ...
    └── scentai_catalog.sqlite3
```

Gerekli varliklar:

- `chroma_db_bge_m3/`: BGE-M3 ile olusturulmus Chroma veritabani klasorunun
  tamami. Yalnizca `chroma.sqlite3` dosyasini yuklemek yeterli degildir.
- `scentai_catalog.sqlite3`: canonical isimler ve community similarity graph
  iceren SQLite katalogu.

Bu iki varlik Drive'da zaten bu konumlardaysa tekrar yukleme yapma.

## 3. Colab runtime ayari

Colab menusunden:

```text
Runtime > Change runtime type > Hardware accelerator > None
```

CPU runtime yeterlidir. T4 veya A100 secmek bu asamada fayda saglamaz.

Mumkunse onceki notebook'lardan kalma paketlerin bulunmadigi yeni bir Colab
oturumu kullan.

## 4. Calistirma

Notebook acildiktan sonra:

```text
Runtime > Run all
```

Huceleri tek tek degistirme veya atlama. Notebook sirayla:

1. Izole bir `uv` Python ortami kurar.
2. CPU-only retrieval kutuphanelerini yukler.
3. Dependency ve import kontrollerini yapar.
4. Drive'i baglar.
5. Chroma ve SQLite verilerini daha hizli sorgu icin `/content` altina kopyalar.
6. BGE-M3 modelini CPU'da yukler.
7. Retrieval HTTP servisini `127.0.0.1:8020` adresinde baslatir.
8. Isim cozumleme, marka filtresi, negatif filtre ve benzerlik testlerini yapar.
9. Sicak sorgu benchmark'ini calistirir.

Ilk calistirmada kutuphane kurulumu, 1.5 GB civarindaki Chroma klasorunun
kopyalanmasi ve BGE-M3 indirilmesi nedeniyle bekleme olabilir.

## 5. Basarili sonuc

Asagidaki satiri gorursen ana API sozlesme testi basarilidir:

```text
CLEAN RETRIEVAL CONTRACT TEST: PASSED
```

Notebook ayrica su raporu Drive'a kaydeder:

```text
MyDrive/Perfume-Dataset/runs/clean_retrieval_stage2_report.json
```

Stage 2 bittikten sonra kontrol icin bu JSON dosyasi yeterlidir. Colab'in tum
cell output'unu kopyalamak zorunda degilsin. Bir hata olursa yalnizca hata veren
hucenin tam traceback'ini paylas.

## 6. Bu asamada yapilmayacaklar

- Inference notebook'unu ayni anda calistirma.
- vLLM, Unsloth, PEFT veya CUDA paketi kurma.
- Retrieval ortaminda GPU PyTorch kullanma.
- Chroma klasorunun icinden dosya silme.
- Notebook'taki pinned dependency surumlerini degistirme.

Stage 2 raporu temiz ciktiginda sonraki adim, inference ve retrieval servislerini
ayri Python ortamlarinda tutarak ayni Colab runtime icinde baglayan Stage 3
pipeline'i olacaktir.

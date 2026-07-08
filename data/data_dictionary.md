# Data Dictionary (Kamus Data)
Dokumen ini berisi penjelasan mengenai struktur kolom pada dataset yang digunakan dalam project UAS Pemrosesan Bahasa Alami: ABSA dan NER untuk Review E-Commerce.

## 1. Dataset ABSA (`data/processed/balanced.csv`)
Dataset ini digunakan untuk melatih model klasifikasi sentimen berbasis aspek (Aspect-Based Sentiment Analysis). Dataset telah melalui proses *Random Oversampling* untuk mengatasi *data imbalance*.

| Nama Kolom | Tipe Data | Deskripsi | Contoh |
| :--- | :--- | :--- | :--- |
| `review_text` | String | Teks ulasan asli dari pengguna platform e-commerce (Tokopedia) yang telah dibersihkan dari nilai kosong (NaN). | *"Kualitas barang sangat bagus, tapi sayang pengiriman lambat"* |
| `aspect` | String | Kategori aspek utama yang diekstrak dari teks menggunakan pendekatan rule-based (keyword matching). | `kualitas`, `pengiriman`, `harga`, `kemasan`, `pelayanan` |
| `sentiment_label` | String | Label sentimen biner bawaan dari dataset asli. | `positive`, `negative` |
| `absa_label` | String | Label target klasifikasi gabungan antara aspek dan sentimen. | `kualitas_positive`, `pengiriman_negative` |

## 2. Dataset NER (`data/processed/ner_ready.tsv`)
Data ini digunakan sebagai representasi format BIO Tagging untuk Named Entity Recognition.

| Nama Kolom | Tipe Data | Deskripsi | Contoh |
| :--- | :--- | :--- | :--- |
| `Token` | String | Kata tunggal (unigram) hasil pemecahan teks ulasan. | `sepatu`, `murah`, `jne` |
| `Tag` | String | Label BIO (Begin, Inside, Outside) dari token tersebut. | `B-PROD`, `O`, `B-ORG` |

**Daftar Label BIO yang Digunakan:**
* `B-ASPECT`: Penanda awal kata yang merujuk pada aspek produk (contoh: harga, kualitas).
* `B-PROD`: Penanda awal kata yang merujuk pada produk (contoh: sepatu, tas).
* `B-ORG`: Penanda awal kata yang merujuk pada entitas organisasi/ekspedisi (contoh: JNE, Tokopedia).
* `O`: (Outside) Penanda kata yang bukan merupakan entitas penting.
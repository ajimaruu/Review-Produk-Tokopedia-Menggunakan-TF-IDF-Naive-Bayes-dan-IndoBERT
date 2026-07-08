# UAS-NLP-ABSA-NER

Aspect-Based Sentiment Analysis (ABSA) & Named Entity Recognition (NER) pada review produk Tokopedia — menggunakan **TF-IDF + Multinomial Naive Bayes** dan **IndoBERT**, dengan aplikasi demo berbasis **Streamlit**.

| | |
|---|---|
| **Penulis** | Aji Bayu Seno (A11.2023.14885) |
| **Institusi** | Fakultas Ilmu Komputer, Universitas Dian Nuswantoro |
| **Dataset** | ± 3.209 ulasan produk Tokopedia |
| **Akurasi IndoBERT** | 78,91% |
| **Akurasi Naive Bayes** | 72,90% |

---

## Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Struktur Folder](#struktur-folder)
- [Panduan Instalasi](#panduan-instalasi)
- [Alur & Cara Training](#alur--cara-training)
- [Cara Menjalankan Aplikasi](#cara-menjalankan-aplikasi)
- [Contoh Penggunaan](#contoh-penggunaan)
- [Evaluasi Model](#evaluasi-model)
- [Troubleshooting](#troubleshooting)
- [Referensi](#referensi)

---

## Fitur Utama

- **Single Prediction** — analisis satu ulasan: sentimen, confidence score, aspek terdeteksi, dan hasil NER.
- **Batch Prediction** — upload CSV berisi banyak ulasan, diproses sekaligus, hasil dapat diunduh.
- **Dashboard Analitik** — visualisasi distribusi sentimen, distribusi aspek, dan confidence distribution.
- **Dua model sentimen** — Naive Bayes (TF-IDF) dan IndoBERT (fine-tuned).
- **NER berbasis kamus aspek + BIO Tagging** — mengenali aspek Harga, Pengiriman, Kemasan, Kualitas, dan Pelayanan.

---

## Struktur Folder

```
UAS-NLP-ABSA-NER/
│
├── app/                                    # Aplikasi Streamlit
│   ├── pages/
│   │   ├── 1_Single_Prediction.py
│   │   ├── 2_Batch_Prediction.py
│   │   ├── 3_Dashboard.py
│   │   └── 4_About.py
│   ├── styles/
│   │   └── style.css                       # Custom CSS tampilan aplikasi
│   └── app.py                              # Entry point utama
│
├── data/
│   ├── processed/                          # Hasil pipeline preprocessing/training
│   │   ├── absa_ready.csv                  # Dataset final siap dipakai model ABSA
│   │   ├── balanced.csv                    # Dataset setelah balancing kelas sentimen
│   │   ├── ner_ready.tsv                   # Dataset NER siap pakai (format BIO)
│   │   ├── preprocessed.csv                # Hasil case folding/cleaning/normalisasi/stemming
│   │   └── similarity.csv                  # Hasil pencocokan kata via edit distance
│   └── raw/
│       ├── absa_dataset.csv                # Dataset mentah ulasan Tokopedia
│       ├── aspect_dictionary.csv           # Kamus kata/frasa per kategori aspek
│       ├── data_dictionary.md              # Dokumentasi kolom dataset
│       ├── ner_dataset.tsv                 # Dataset mentah berlabel BIO untuk NER
│       └── normalization.csv               # Kamus normalisasi kata tidak baku → baku
│
├── demo/                                   # Aset/berkas pendukung demo aplikasi
│
├── models/
│   ├── bert/                               # Output fine-tuning IndoBERT (Hugging Face)
│   │   ├── checkpoint-323/
│   │   ├── checkpoint-625/
│   │   ├── checkpoint-646/
│   │   ├── checkpoint-969/
│   │   ├── checkpoint-1292/
│   │   ├── checkpoint-1615/                # Checkpoint per step training
│   │   ├── config.json
│   │   ├── model.safetensors               # Bobot model IndoBERT final
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   ├── training_args.bin
│   │   ├── vocab.txt
│   │   ├── absa_nb_model.pkl               # (salinan) model Naive Bayes
│   │   └── tfidf_vectorizer.pkl            # (salinan) vectorizer TF-IDF
│   ├── absa_nb_model.pkl                   # Model Multinomial Naive Bayes terlatih
│   ├── tfidf_matrix.pkl                    # Matriks TF-IDF hasil fit pada data training
│   └── tfidf_vectorizer.pkl                # TfidfVectorizer terlatih
│
├── notebooks/
│   └── eksperimen_absa_ner.ipynb           # Notebook eksplorasi & eksperimen model
│
├── presentation/
│   └── Presentasi_ABSA_NER_Tokopedia.pdf   # Slide presentasi penelitian
│
├── proof/                                  # Bukti tangkapan layar aplikasi berjalan
│   ├── screenshot_ui_batch_prediction.png
│   ├── screenshot_ui_dashboard.png
│   └── screenshot_ui_single_prediction.png
│
├── reports/
│   ├── app.log                             # Log runtime aplikasi
│   ├── confusion_matrix.png                # Confusion matrix hasil evaluasi
│   ├── error_analysis.csv                  # Daftar prediksi salah untuk analisis
│   ├── error_analysis.md                   # Ringkasan analisis kesalahan model
│   ├── Laporan_UAS_ABSA_NER_Tokopedia_TFIDF_NaiveBayes_IndoBERT.pdf
│   ├── metrics.json                        # Ringkasan metrik evaluasi (accuracy, precision, recall, F1)
│   └── prediction_result.csv               # Contoh hasil prediksi batch
│
├── src/                                    # Kode sumber pipeline (CLI scripts)
│   ├── __pycache__/
│   ├── balance_dataset.py                  # Menyeimbangkan jumlah data per kelas sentimen
│   ├── edit_distance.py                    # Utility pencocokan kata (Levenshtein) untuk normalisasi
│   ├── evaluate_bert.py                    # Evaluasi model IndoBERT (metrics, confusion matrix)
│   ├── ner_predict.py                      # Prediksi/ekstraksi aspek (NER) dari teks
│   ├── predict.py                          # Prediksi ABSA end-to-end (sentimen + aspek)
│   ├── prepare_dataset.py                  # Menyiapkan/menggabungkan dataset mentah
│   ├── preprocessing.py                    # Case folding, cleaning, normalisasi, stopword, stemming
│   ├── train_absa.py                       # Training TF-IDF + Multinomial Naive Bayes
│   ├── train_bert.py                       # Fine-tuning IndoBERT
│   └── train_ner.py                        # Membangun kamus aspek / data BIO Tagging
│
├── tmp_trainer/                            # Folder sementara Hugging Face Trainer (aman diabaikan/di-gitignore)
├── venv/                                   # Virtual environment (di-gitignore)
├── README.md
└── requirements.txt
```

> `models/bert/absa_nb_model.pkl` dan `models/bert/tfidf_vectorizer.pkl` adalah salinan dari file yang sama di `models/`. Jika ingin merapikan repo, cukup simpan satu salinan saja di `models/`.

---

## Panduan Instalasi

### 1. Prasyarat

- Python **3.9 – 3.11**
- pip / virtualenv
- (Opsional, direkomendasikan) GPU dengan CUDA untuk mempercepat fine-tuning IndoBERT di `src/train_bert.py`
- Git

### 2. Clone Repository

```bash
git clone https://github.com/<username>/UAS-NLP-ABSA-NER.git
cd UAS-NLP-ABSA-NER
```

### 3. Buat & Aktifkan Virtual Environment

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Contoh isi `requirements.txt`:**

```txt
streamlit>=1.32
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
Sastrawi>=1.0.1
nltk>=3.8
transformers>=4.38
torch>=2.1
plotly>=5.20
tqdm>=4.66
python-Levenshtein>=0.25
```

### 5. Download Resource NLTK (jika `preprocessing.py` menggunakan stopword NLTK)

```bash
python -m nltk.downloader stopwords punkt
```

---

## Alur & Cara Training

Seluruh proses berikut dijalankan dari **root folder proyek**, menggunakan script di `src/`. Jalankan berurutan sesuai tahapan pipeline:

### 1. Menyiapkan Dataset Mentah

```bash
python src/prepare_dataset.py
```

Membaca `data/raw/absa_dataset.csv` dan `data/raw/ner_dataset.tsv`, membersihkan baris kosong/NaN, lalu menyiapkan dataset kerja awal.

### 2. Text Preprocessing

```bash
python src/preprocessing.py
```

Menjalankan tahapan **case folding → cleaning → normalisasi → stopword removal → stemming**. Proses normalisasi memanfaatkan `data/raw/normalization.csv` dan modul `src/edit_distance.py` untuk mencocokkan kata tidak baku (menghasilkan `data/processed/similarity.csv`). Output akhir: `data/processed/preprocessed.csv`.

### 3. Balancing Dataset

```bash
python src/balance_dataset.py
```

Menyeimbangkan jumlah data per kelas sentimen (Positive/Neutral/Negative) agar model tidak bias ke kelas mayoritas. Output: `data/processed/balanced.csv`.

### 4. Training Naive Bayes (TF-IDF)

```bash
python src/train_absa.py
```

Melatih `TfidfVectorizer` + `MultinomialNB` pada `data/processed/balanced.csv`. Output:
- `models/absa_nb_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/tfidf_matrix.pkl`
- `data/processed/absa_ready.csv`

Contoh output terminal:

```
Jumlah data setelah dibersihkan : 3209
Melakukan TF-IDF...
TF-IDF selesai.
Data Train : 2567
Data Test  : 642

Training Naive Bayes...
Training selesai.

==========================================
HASIL EVALUASI MODEL NAIVE BAYES
==========================================
Accuracy  : 72.90%
Precision : 72.50%
Recall    : 72.90%
F1-score  : 72.33%
```

### 5. Membangun Kamus Aspek & Data NER (BIO Tagging)

```bash
python src/train_ner.py
```

Menyusun/memperbarui `data/raw/aspect_dictionary.csv` dan menghasilkan `data/processed/ner_ready.tsv` (token + label BIO: `B-ASPECT` / `O`) yang dipakai oleh `src/ner_predict.py` saat runtime.

### 6. Fine-tuning IndoBERT

```bash
python src/train_bert.py
```

Melakukan fine-tuning `indobenchmark/indobert-base-p1` menggunakan Hugging Face `Trainer` pada `data/processed/balanced.csv`. Checkpoint otomatis tersimpan di `models/bert/checkpoint-<step>/`, sedangkan model final (`config.json`, `model.safetensors`, tokenizer, `training_args.bin`) tersimpan langsung di `models/bert/`.

> Jika ingin mengubah jumlah epoch, batch size, atau learning rate, sesuaikan parameter/argumen di dalam `src/train_bert.py`.

### 7. Evaluasi Model IndoBERT

```bash
python src/evaluate_bert.py
```

Menghasilkan:
- `reports/metrics.json` — accuracy, precision, recall, F1-score
- `reports/confusion_matrix.png`
- `reports/error_analysis.csv` & `reports/error_analysis.md`

---

## Cara Menjalankan Aplikasi

Pastikan langkah training di atas sudah selesai (atau model sudah tersedia di `models/` & `models/bert/`), lalu jalankan dari root folder:

```bash
streamlit run app/app.py
```

Aplikasi terbuka otomatis di:

```
http://localhost:8501
```

### Menjalankan pada port/host tertentu

```bash
streamlit run app/app.py --server.port 8080 --server.address 0.0.0.0
```

### Menjalankan dengan Docker (opsional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]
```

```bash
docker build -t uas-nlp-absa-ner .
docker run -p 8501:8501 uas-nlp-absa-ner
```

---

## Contoh Penggunaan

### 1. Single Prediction — via UI

1. Buka menu **Single Prediction** (`app/pages/1_Single_Prediction.py`) di sidebar.
2. Pilih model: **IndoBERT** atau **Naive Bayes**.
3. Masukkan teks ulasan, misalnya:
   ```
   packing bagus tapi pengiriman lambat
   ```
4. Klik **Analyze**.

**Contoh hasil:**

| Field | Nilai |
|---|---|
| Sentiment | Neutral |
| Confidence | 99.94% |
| Prediction Time | 0.060s |
| Aspect | Pengiriman, Kemasan |

| Token | Label |
|---|---|
| packing | B-ASPECT |
| bagus | O |
| tapi | O |
| pengiriman | B-ASPECT |
| lambat | O |

### 2. Single Prediction — via CLI (`src/predict.py`)

```bash
python src/predict.py --text "packing bagus tapi pengiriman lambat" --model indobert
```

Contoh output:

```json
{
  "original_text": "packing bagus tapi pengiriman lambat",
  "preprocessed_text": "packing bagus kirim lambat",
  "aspect": ["Pengiriman", "Kemasan"],
  "entities": [
    {"token": "packing", "label": "B-ASPECT"},
    {"token": "pengiriman", "label": "B-ASPECT"}
  ],
  "sentiment": "Neutral",
  "confidence": 99.94,
  "prediction_time": 0.06,
  "model": "IndoBERT"
}
```

### 3. Menguji NER Saja (`src/ner_predict.py`)

```bash
python src/ner_predict.py --text "pengiriman sangat cepat"
```

```
Token       Label
----------  ---------
pengiriman  B-ASPECT
sangat      O
cepat       O
```

### 4. Batch Prediction — via UI

1. Buka menu **Batch Prediction** (`app/pages/2_Batch_Prediction.py`).
2. Siapkan file CSV dengan minimal kolom `text`, contoh:

   ```csv
   text
   "Dikemas dengan baik"
   "Barang sesuai deskripsi dan pesanan, packing super aman dan rapi"
   "Sampe di rmh saya, kurir lgs minta maaf di depan rumah"
   ```

3. Upload file, pilih model, klik **Mulai Prediksi**.
4. Unduh hasil melalui tombol **Download Result**.

**Contoh ringkasan hasil (501 data):**

| Metrik | Nilai |
|---|---|
| Positive | 190 |
| Negative | 157 |
| Neutral | 154 |
| Avg. Confidence | 98.60% |
| Waktu eksekusi | 6.31 detik |

Hasil batch juga otomatis tersimpan sebagai contoh di `reports/prediction_result.csv`.

### 5. Dashboard Analitik

Buka menu **Dashboard** (`app/pages/3_Dashboard.py`) untuk melihat:

- Distribusi sentimen (donut chart & bar chart)
- Distribusi aspek (Pengiriman, Kualitas, Penjual, Pelayanan, Kemasan, Harga, Keaslian)
- Confidence distribution
- Filter hasil berdasarkan sentimen + unduh CSV

---

## Evaluasi Model

Ringkasan tersimpan otomatis di `reports/metrics.json` dan `reports/confusion_matrix.png` setelah menjalankan `src/evaluate_bert.py`.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes (TF-IDF) | 72.90% | 72.50% | 72.90% | 72.33% |
| IndoBERT | **78.91%** | **79.64%** | **78.96%** | **79.22%** |

**Ringkasan:**
- **IndoBERT** lebih unggul memahami konteks bahasa Indonesia (termasuk negasi & makna implisit) berkat arsitektur bidirectional transformer.
- **Naive Bayes** lebih unggul dari sisi kecepatan proses dan kebutuhan resource, cocok untuk infrastruktur terbatas atau kebutuhan real-time.
- Detail baris prediksi yang salah dapat dicek di `reports/error_analysis.csv` / `reports/error_analysis.md`.

---

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `ModuleNotFoundError: Sastrawi` | Jalankan `pip install Sastrawi` |
| Model IndoBERT gagal dimuat (`OSError`) | Pastikan folder `models/bert/` berisi `config.json`, `model.safetensors`, `tokenizer.json`, `vocab.txt`, `special_tokens_map.json` |
| Prediksi IndoBERT sangat lambat | Jalankan di mesin ber-GPU, atau gunakan model Naive Bayes (`models/absa_nb_model.pkl`) untuk kebutuhan cepat |
| Streamlit error `Port already in use` | Ganti port: `streamlit run app/app.py --server.port 8502` |
| Upload CSV gagal (`KeyError: 'text'`) | Pastikan file CSV memiliki kolom bernama persis `text` |
| Aspek terdeteksi kosong/`None` | Perluas `data/raw/aspect_dictionary.csv` dengan frasa aspek baru, lalu jalankan ulang `src/train_ner.py` |
| Checkpoint `models/bert/checkpoint-*` memenuhi disk | Aman dihapus setelah training selesai; yang dibutuhkan aplikasi hanya file model final di root `models/bert/` |
| Folder `tmp_trainer/` menumpuk saat training ulang | Boleh dihapus sebelum menjalankan `src/train_bert.py` lagi |

---


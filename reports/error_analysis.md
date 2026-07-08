# Error Analysis

| Text | Actual | Prediction | Penyebab |
|------|--------|------------|----------|
| hati2 belanja disini, nomor tidak sesuai ukuran kaki | Negative | Positive | Model lebih banyak menangkap kata netral/positif dibanding frasa "tidak sesuai", sehingga gagal mengenali sentimen negatif. |
| barang diterima sesuai pesanan. ekspedisi lancar dan packing aman sampai tujuan. setelah dicoba, kualitas product masih belum terbukti secara ampuh mengusir cicak. | Neutral | Positive | Review memiliki sentimen campuran. Awal kalimat positif, sedangkan akhir menyampaikan kekecewaan terhadap kualitas produk. |
| beli 48 pack... barang bocor..... respon penjual buruk... packing buruk... kurir buruk... | Negative | Positive | Banyak kata negatif, namun model gagal memahami konteks keseluruhan sehingga salah mengklasifikasikan menjadi positif. |
| pengiriman oke, dan untuk scooternya ternyata gk bisa dibelokan dari stang. kalau untuk kualitas ya sesuai dengan harga, yang buat kecewa ya ternyata tidak bisa dibelokin stangnya | Negative | Positive | Terdapat aspek positif (pengiriman) dan negatif (fungsi produk), sehingga model lebih fokus pada kata-kata positif. |
| Produk : Bahan agak tipis. Pengiriman : Good. Pengemasan : Poor. Penjual : Good. Harga : Standard | Neutral | Positive | Review berisi penilaian beberapa aspek dengan polaritas berbeda. Model cenderung memilih kelas positif karena terdapat kata "Good". |

## Kesimpulan

Berdasarkan hasil evaluasi model, beberapa penyebab utama kesalahan klasifikasi adalah:

1. **Review memiliki lebih dari satu aspek** dengan sentimen yang berbeda (misalnya produk positif tetapi pengiriman negatif).
2. **Model TF-IDF + Naive Bayes tidak memahami konteks kalimat**, hanya mengandalkan frekuensi kata.
3. **Kata negasi** seperti *tidak*, *belum*, atau *kurang* belum ditangani dengan baik sehingga makna kalimat berubah.
4. **Campuran kata positif dan negatif** dalam satu review membuat model cenderung memilih kelas yang memiliki kata dengan bobot lebih tinggi.
5. Model belum mampu memahami hubungan antar kata (context-aware), sehingga performanya terbatas pada representasi berbasis bag-of-words/TF-IDF.
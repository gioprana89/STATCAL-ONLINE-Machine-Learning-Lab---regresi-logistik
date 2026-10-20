# STATCAL ONLINE — Machine Learning Lab
## Praktikum 4 Pertanyaan Regresi Logistik (Bab 1)

Ini adalah **versi modifikasi** aplikasi yang mengikuti empat pertanyaan praktikum pertama:

1. **True Positive (TP):** Berapa mahasiswa yang diprediksi Tidak Lulus dengan benar?
2. **False Negative (FN):** Berapa mahasiswa yang sebenarnya Tidak Lulus tetapi keliru diprediksi Lulus?
3. **Recall:** Seberapa baik model mengenali mahasiswa berisiko Tidak Lulus?
4. **Probabilitas Profil:** Berapa probabilitas seorang mahasiswa diprediksi Tidak Lulus?

Empat jawaban tersedia pada **tab terpisah**. Ada tab *Kasus & Data* untuk melihat data training/testing dan tab *Laporan Excel* untuk mengunduh seluruh hasil.

## Isi paket
- `app.py`: antarmuka aplikasi Streamlit
- `ml_core.py`: pemodelan regresi logistik dan generator file Excel
- `dataset_mahasiswa.csv`: dataset sintetis 1.000 mahasiswa
- `requirements.txt`: dependensi

## Jalankan di komputer Windows

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Letakkan semua file dalam folder yang **sama**. Pilih `app.py` sebagai file utama saat deploy ke Streamlit Community Cloud, dan unggah `ml_core.py` serta `dataset_mahasiswa.csv` dalam repository yang sama.

## Pengaturan awal untuk mengikuti materi buku
- Target (Y): `Kelulusan_MK`
- Kelas positif (1): `Tidak Lulus`
- Variabel X: seluruh 8 variabel prediktor yang tersedia
- Data training 80%, testing 20%
- Threshold 0,50
- C 1,0, bobot kelas normal, random seed 42

## Cara menggunakan
1. Jika mengunggah file `.xlsx` dengan banyak sheet, pilih **DATA_MAHASISWA** (secara otomatis dipilih jika ada).
2. Buka **Kasus & Data** untuk melihat data lengkap, training, dan testing.
3. Buka **Q1**, **Q2**, dan **Q3** untuk melihat TP, FN, serta Recall berdasarkan data testing.
4. Buka **Q4**, masukkan contoh profil lalu klik **Hitung Probabilitas Profil**.
5. Terakhir, masuk ke **Laporan Excel**, lalu unduh file `.xlsx` yang berisi hasil semua pertanyaan dan data training/testing lengkap.

## Isi workbook Excel (11 worksheet)
1. `Ringkasan` — hasil inti dan informasi konfigurasi.
2. `Pertanyaan & Jawaban` — jawaban numerik dan interpretasi 4 pertanyaan.
3. `Pengaturan Model` — konfigurasi split dan model.
4. `Data Lengkap` — semua observasi serta penanda training/testing.
5. `Data Training` — dataset yang dipakai melatih model.
6. `Data Testing` — dataset yang dipakai mengevaluasi model **beserta prediksi**.
7. `Prediksi Testing` — tabel ringkas probabilitas, prediksi, dan TP/TN/FP/FN.
8. `Metrik Evaluasi` — accuracy, precision, recall, F1-score, ROC-AUC, dan baseline.
9. `Simulasi Profil` — input contoh dan probabilitas hasil Q4 (jika sudah dijalankan).
10. `Kamus Variabel` — deskripsi variabel.
11. `Confusion Matrix` — matriks aktual vs prediksi.

**Catatan penting:** data training ditampilkan dalam bentuk aslinya, bukan fitur numerik hasil encoding/standardisasi. Prediksi dan nilai evaluasi dihitung pada data testing. Jika belum menjalankan Q4, sel hasil simulasi profil dalam laporan akan menunjukkan status “Belum dijalankan”.

Dataset bawaan **sintetis** dan untuk keperluan pembelajaran saja. Prediksi bukan kepastian atau landasan keputusan akademik terhadap mahasiswa sungguhan.

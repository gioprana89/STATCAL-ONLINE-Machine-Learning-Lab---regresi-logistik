# STATCAL ONLINE — Machine Learning Lab: Logistic Regression

**Praktikum 1: Prediksi Kelulusan Mata Kuliah Mahasiswa (data sintetis).**

## Struktur paket

- `app.py` — seluruh kode aplikasi Streamlit.
- `dataset_mahasiswa.csv` — dataset simulasi 1.000 mahasiswa yang digunakan pada contoh.
- `requirements.txt` — dependensi untuk Streamlit Community Cloud atau lokal.

## Cara menjalankan (Windows / komputer lokal)

1. Pasang Python versi 3.10 atau lebih baru.
2. Buka Terminal/Command Prompt di folder paket.
3. Jalankan `python -m pip install -r requirements.txt`.
4. Jalankan `python -m streamlit run app.py`.
5. Buka URL lokal yang ditampilkan Streamlit (umumnya `http://localhost:8501`).

## Cara deploy di Streamlit Community Cloud

1. Buat repository GitHub dan unggah tiga file di atas ke root repository.
2. Di Streamlit Community Cloud, pilih repository dan file utama `app.py`.
3. Deploy. Aplikasi otomatis memakai `dataset_mahasiswa.csv` dan juga menerima upload CSV/XLSX.

## Tujuan belajar

1. Mengenali dataset dan variabel X/Y.
2. Memahami pembagian training dan testing yang terstratifikasi.
3. Melatih Logistic Regression menggunakan pipeline preprocessing (imputasi, standardisasi, dan one-hot encoding).
4. Mengevaluasi accuracy, precision, recall, F1, ROC-AUC, dan confusion matrix.
5. Mengubah threshold dan membandingkan hasil.
6. Mengisi lembar pertanyaan investigasi dan mengunduh jawaban.

## Catatan metodologis

- Default kelas positif ialah **Tidak Lulus** agar recall dapat diinterpretasikan sebagai proporsi mahasiswa berisiko yang berhasil dikenali model.
- ID mahasiswa tidak digunakan sebagai fitur model.
- Semua transformasi data dipelajari dari **training set** melalui scikit-learn `Pipeline`.
- Evaluasi memakai satu hold-out stratified split. Hasil berubah jika pengaturan dan ukuran data testing berubah; angka tidak otomatis bisa dibandingkan secara kausal.
- Probabilitas klasifikasi tidak merupakan kepastian; threshold mengubah label prediksi, bukan peluang yang sudah dihasilkan model.
- Data sintetis ini hanya untuk pembelajaran; jangan gunakan untuk keputusan nyata mengenai mahasiswa.

## Referensi teknis

- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- https://docs.streamlit.io/develop/api-reference

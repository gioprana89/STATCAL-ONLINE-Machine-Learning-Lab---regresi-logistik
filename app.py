"""STATCAL ONLINE – Praktikum Machine Learning: Regresi Logistik.

Dataset awal: 1.000 mahasiswa (DATA SINTETIS). Aplikasi bisa menerima CSV/XLSX sendiri.
Jalankan dengan: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="ML Lab | Regresi Logistik", page_icon="🎓", layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
div[data-testid="stMetric"] {background:#f3f7fb; border-radius:10px; padding:12px;}
h1 {font-size:2rem!important;}
.small-note {color:#607080; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "dataset_mahasiswa.csv"
DEFAULT_TARGET = "Kelulusan_MK"
ID_COLUMNS = {"ID_Mahasiswa"}
GUIDE = {
    "Jam_Belajar_Mingguan": "Jumlah jam belajar setiap minggu.",
    "Kehadiran_Persen": "Persentase kehadiran perkuliahan.",
    "Nilai_Tugas_Awal": "Nilai tugas yang tersedia sebelum akhir semester.",
    "Nilai_Kuis_Awal": "Nilai kuis yang tersedia sebelum akhir semester.",
    "Aktivitas_Belajar": "Indeks aktivitas belajar (skala 1–5 pada dataset awal).",
    "Akses_LMS_Mingguan": "Frekuensi mengakses learning management system per minggu.",
    "Tugas_Tepat_Waktu_Persen": "Persentase tugas dikumpulkan tepat waktu.",
    "Metode_Belajar": "Kategori cara belajar: Mandiri, Kelompok, atau Campuran.",
    "Kelulusan_MK": "Hasil akhir: Lulus atau Tidak Lulus; variabel target (Y).",
    "ID_Mahasiswa": "Kode identifikasi; tidak digunakan untuk melatih model."
}
QUESTIONS = [
    ("Eksplorasi", "Berapa jumlah dan persentase mahasiswa Lulus serta Tidak Lulus?", "Lihat grafik distribusi kelas pada tab Data."),
    ("Eksplorasi", "Apakah rata-rata jam belajar berbeda antara mahasiswa Lulus dan Tidak Lulus?", "Lihat ringkasan kelompok dan grafik fitur."),
    ("Eksplorasi", "Bagaimana rata-rata kehadiran mahasiswa menurut kelompok kelulusan?", "Pilih Kehadiran_Persen pada tab Data."),
    ("Penerapan", "Berapa observasi yang digunakan sebagai training dan testing jika pembagian 80:20?", "Lihat jumlah data pelatihan dan pengujian di tab Model."),
    ("Penerapan", "Berapakah accuracy, precision, recall, dan F1-score model?", "Lihat metrik pada tab Evaluasi; perhatikan kelas positif yang dipilih."),
    ("Evaluasi", "Berapa banyak mahasiswa sebenarnya Tidak Lulus tetapi keliru diprediksi Lulus?", "Jadikan Tidak Lulus sebagai kelas positif, lalu cari FN pada confusion matrix."),
    ("Evaluasi", "Apakah akurasi model mengungguli prediksi kelas mayoritas saja?", "Bandingkan dengan baseline pada tab Evaluasi."),
    ("Simulasi", "Bagaimana jumlah false negative berubah saat threshold diturunkan?", "Geser ambang keputusan (threshold), mis. dari 0,50 menjadi 0,30."),
    ("Simulasi", "Apa yang berubah ketika proporsi training 70%, 80%, dan 90%?", "Ubah persentase training; bandingkan metrik dengan mempertimbangkan data testing yang berubah."),
    ("Simulasi", "Apakah hasil berubah jika Nilai_Kuis_Awal tidak digunakan sebagai X?", "Hilangkan fitur tersebut di panel samping dan latih kembali."),
    ("Interpretasi", "Variabel apa yang koefisien modelnya paling besar secara absolut?", "Gunakan tab Evaluasi → Koefisien; bedakan asosiasi model dengan sebab-akibat."),
    ("Interpretasi", "Untuk profil mahasiswa contoh yang Anda masukkan, berapa probabilitas kelas positif?", "Masukkan skenario ke tab Simulasi Profil."),
]

@st.cache_data(show_spinner=False)
def load_default(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def format_percent(v: float) -> str:
    return f"{v * 100:.1f}%"


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str], C: float, balanced: bool):
    transformers = []
    if numeric_cols:
        transformers.append(("numerik", Pipeline([
            ("isi_kosong", SimpleImputer(strategy="median")),
            ("standar", StandardScaler())
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("kategori", Pipeline([
            ("isi_kosong", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]), categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        C=C, max_iter=2000, solver="lbfgs", class_weight="balanced" if balanced else None
    )
    return Pipeline([("persiapan", preprocessor), ("regresi_logistik", classifier)])


def bar_categories(df: pd.DataFrame, target: str) -> None:
    counts = df[target].astype(str).value_counts(dropna=False)
    st.bar_chart(counts.rename("Jumlah mahasiswa"))


def plot_confusion(cm: np.ndarray, negative_name: str, positive_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=17, fontweight="bold")
    ax.set_xticks([0, 1], [negative_name, positive_name])
    ax.set_yticks([0, 1], [negative_name, positive_name])
    ax.set_xlabel("Prediksi model")
    ax.set_ylabel("Kelas sebenarnya")
    ax.set_title("Confusion Matrix — Data Testing")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: pd.Series, probabilities: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(fpr, tpr, linewidth=2.2, label=f"Logistic Regression (AUC={roc_auc_score(y_true, probabilities):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Tebakan acak")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Data Testing")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig

st.title("🎓 Machine Learning Lab: Regresi Logistik")
st.caption("Kasus → Dataset → Aplikasi → Evaluasi | Bahan praktikum interaktif untuk pemula")
st.info("**Tujuan:** memahami bagaimana model memperkirakan peluang kelulusan mata kuliah dari fitur belajar mahasiswa. Seluruh data bawaan bersifat **sintetis**, bukan data mahasiswa sebenarnya.")

with st.sidebar:
    st.header("⚙️ Pengaturan Praktikum")
    uploaded = st.file_uploader("Unggah dataset sendiri (opsional)", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.success("Dataset berhasil dibaca.")
        except Exception as err:
            st.error(f"File tidak dapat dibaca: {err}")
            st.stop()
    elif DEFAULT_DATA.exists():
        df = load_default(str(DEFAULT_DATA))
        st.caption("Menggunakan dataset simulasi 1.000 mahasiswa yang disertakan.")
    else:
        st.warning("Unggah file CSV/XLSX terlebih dahulu; dataset bawaan tidak ditemukan.")
        st.stop()

    if df.empty or df.shape[1] < 2:
        st.error("Dataset harus memiliki baris dan sedikitnya dua kolom.")
        st.stop()
    df = df.copy()
    st.markdown("---")
    all_targets = list(df.columns)
    target = st.selectbox("Target klasifikasi (Y)", all_targets,
                          index=all_targets.index(DEFAULT_TARGET) if DEFAULT_TARGET in all_targets else len(all_targets)-1,
                          help="Target harus mempunyai dua kelas. Untuk dataset bawaan: Kelulusan_MK.")
    valid = df[target].notna()
    df = df.loc[valid].copy()
    target_classes = sorted(df[target].astype(str).unique().tolist())
    if len(target_classes) != 2:
        st.error("Regresi logistik pada praktikum ini memerlukan target dengan tepat 2 kategori berbeda.")
        st.stop()

    positive_default = "Tidak Lulus" if "Tidak Lulus" in target_classes else target_classes[-1]
    pos = st.selectbox("Kelas positif (nilai 1)", target_classes,
                       index=target_classes.index(positive_default),
                       help="Metrik precision/recall/F1 mengukur kemampuan mengenali kelas positif, misalnya mahasiswa berisiko tidak lulus.")
    neg = next(label for label in target_classes if label != pos)

    exclude = [target] + [c for c in df.columns if c in ID_COLUMNS]
    candidate = [c for c in df.columns if c not in exclude]
    if not candidate:
        st.error("Tidak ada prediktor (X) yang tersedia.")
        st.stop()
    chosen = st.multiselect("Pilih variabel prediktor (X)", options=candidate,
                            default=candidate,
                            help="Hindari ID dan variabel yang baru diketahui setelah hasil kelulusan tersedia.")
    if not chosen:
        st.warning("Pilih minimal satu fitur X untuk melatih model.")
        st.stop()
    train_percent = st.slider("Persentase data training", 60, 90, 80, 5,
                              help="Data lainnya menjadi testing. Pembagian berdasarkan kelas (stratified).")
    st.markdown("### Eksperimen")
    threshold = st.slider("Threshold prediksi positif", 0.10, 0.90, 0.50, 0.05,
                          help="Jika probabilitas kelas positif ≥ threshold, prediksi = kelas positif.")
    with st.expander("Pengaturan lanjut (opsional)"):
        C = st.select_slider("C (kekuatan regularisasi)",
                             options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0,
                             help="Nilai C lebih kecil berarti regularisasi lebih kuat; pemula dapat membiarkan 1.0.")
        balanced = st.checkbox("Seimbangkan bobot kelas", value=False,
                               help="Menyesuaikan bobot berdasarkan frekuensi kelas; dapat memengaruhi precision dan recall.")
        random_state = st.number_input("Random seed", 0, 999999, 42, 1,
                                       help="Seed tetap memudahkan percobaan yang dapat direproduksi.")

# Persiapan dilakukan setelah pembagian data sehingga tidak terjadi data leakage.
X = df[chosen].copy()
for col in chosen:
    if pd.api.types.is_object_dtype(X[col]) or isinstance(X[col].dtype, pd.CategoricalDtype):
        X[col] = X[col].astype("string").replace("", pd.NA).astype(object)
numeric_cols = [c for c in chosen if pd.api.types.is_numeric_dtype(X[c])]
categorical_cols = [c for c in chosen if c not in numeric_cols]
if any(X[col].isna().all() for col in chosen):
    st.error("Ada variabel X yang seluruh nilainya kosong. Hapus variabel tersebut dari pilihan.")
    st.stop()
y = (df[target].astype(str) == pos).astype(int)
if min(y.value_counts()) < 2:
    st.error("Setiap kategori target perlu sedikitnya dua observasi untuk pembagian training/testing.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(100-train_percent)/100, random_state=int(random_state), stratify=y
    )
    model = build_pipeline(numeric_cols, categorical_cols, float(C), balanced)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, list(model.named_steps["regresi_logistik"].classes_).index(1)]
    predicted = (prob >= threshold).astype(int)
except Exception as err:
    st.error(f"Model tidak dapat dijalankan. Periksa dataset dan pengaturan fitur. Detail: {err}")
    st.stop()

score_accuracy = accuracy_score(y_test, predicted)
score_precision = precision_score(y_test, predicted, zero_division=0)
score_recall = recall_score(y_test, predicted, zero_division=0)
score_f1 = f1_score(y_test, predicted, zero_division=0)
auc = roc_auc_score(y_test, prob) if y_test.nunique() == 2 else float("nan")
cm = confusion_matrix(y_test, predicted, labels=[0, 1])
base_class = int(y_train.mean() >= 0.5)
baseline_accuracy = accuracy_score(y_test, np.full(len(y_test), base_class))

st.success(f"Model Logistic Regression berhasil dilatih menggunakan **{len(X_train):,} data training** dan diuji pada **{len(X_test):,} data testing**. Kelas positif: **{pos}**.")
tab_data, tab_model, tab_eval, tab_single, tab_questions = st.tabs([
    "📂 1. Kasus & Data", "⚙️ 2. Model", "📊 3. Evaluasi", "🧪 4. Simulasi Profil", "📝 5. Pertanyaan Praktik"
])

with tab_data:
    st.subheader("Studi kasus: Identifikasi risiko tidak lulus mata kuliah")
    st.write("Sebuah program studi ingin menggunakan informasi belajar yang tersedia **sebelum semester berakhir** untuk memperkirakan peluang kelulusan. Prediksi ini dapat membantu menentukan siapa yang perlu mendapat dukungan belajar tambahan, bukan menjadi dasar keputusan otomatis tentang mahasiswa.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observasi", f"{len(df):,}")
    c2.metric("Jumlah fitur X tersedia", len(candidate))
    c3.metric("Jumlah nilai kosong", f"{int(df.isna().sum().sum()):,}")
    st.markdown("#### Cuplikan dataset")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    with st.expander("Penjelasan variabel dalam dataset awal"):
        dd = pd.DataFrame([{"Variabel": c, "Peran": "Target (Y)" if c == target else "ID" if c in ID_COLUMNS else "Prediktor (X)",
                            "Keterangan": GUIDE.get(c, "Variabel dari file unggahan.")} for c in df.columns])
        st.dataframe(dd, use_container_width=True, hide_index=True)
    st.markdown("#### Distribusi kelas")
    bar_categories(df, target)
    show_col = st.selectbox("Pilih variabel untuk eksplorasi berdasarkan kelas", chosen,
                            index=chosen.index("Jam_Belajar_Mingguan") if "Jam_Belajar_Mingguan" in chosen else 0)
    if show_col in numeric_cols:
        grouped = df.groupby(target)[show_col].agg(["count", "mean", "median", "min", "max"]).round(2)
        st.dataframe(grouped, use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        subgroups = [pd.to_numeric(df.loc[df[target].astype(str)==x, show_col], errors="coerce").dropna() for x in target_classes]
        ax.boxplot(subgroups, tick_labels=target_classes)
        ax.set_ylabel(show_col)
        ax.set_title("Perbandingan distribusi berdasarkan kelulusan")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    else:
        compare = pd.crosstab(df[show_col].astype(str), df[target].astype(str))
        st.dataframe(compare, use_container_width=True)
        st.bar_chart(compare)
    if DEFAULT_DATA.exists() and uploaded is None:
        st.download_button("⬇️ Unduh dataset simulasi (CSV)",
                           data=DEFAULT_DATA.read_bytes(), file_name="Dataset_Machine_Learning_1000_Mahasiswa.csv",
                           mime="text/csv")

with tab_model:
    st.subheader("Pengaturan dan proses pembelajaran model")
    st.markdown("""
**Apa itu Logistic Regression?** Sebuah algoritma **klasifikasi** yang menghitung kemungkinan suatu observasi termasuk salah satu dari dua kelas. Walaupun namanya *regression*, pada praktikum ini tugas utamanya adalah **memprediksi kategori**, bukan nilai numerik.
    """)
    a, b, c = st.columns(3)
    a.metric("Training", f"{len(X_train):,} mahasiswa")
    b.metric("Testing", f"{len(X_test):,} mahasiswa")
    c.metric("Prediktor terpilih", len(chosen))
    st.markdown("#### Tahapan yang dijalankan aplikasi")
    st.markdown("""
1. Membagi data menjadi training dan testing dengan **stratifikasi** agar komposisi target relatif terjaga.
2. Mengisi nilai fitur numerik yang hilang dengan median dari data training, lalu menstandarkan skala numerik.
3. Mengubah fitur kategorik menjadi kolom numerik melalui **one-hot encoding**.
4. Melatih Logistic Regression pada data training **saja**.
5. Memperkirakan probabilitas kelas positif pada data testing yang belum digunakan untuk melatih model.
6. Mengubah probabilitas menjadi prediksi kategori berdasarkan **threshold** yang Anda pilih.
    """)
    st.caption("Proses pengisian, standardisasi, dan encoding dipelajari hanya dari data training menggunakan Pipeline untuk mengurangi risiko kebocoran informasi.")
    st.markdown("#### Fitur yang digunakan")
    st.dataframe(pd.DataFrame({"Fitur X": chosen,
                              "Jenis": ["Numerik" if x in numeric_cols else "Kategorik" for x in chosen],
                              "Keterangan": [GUIDE.get(x, "Variabel dari unggahan") for x in chosen]}),
                 hide_index=True, use_container_width=True)
    st.info("**Pertanyaan untuk mahasiswa:** Mengapa data testing harus dipisahkan dari data training? Mengapa ID_Mahasiswa tidak dimasukkan sebagai prediktor?")

with tab_eval:
    st.subheader("Evaluasi model pada data testing")
    st.caption(f"Semua metrik klasifikasi di bawah menggunakan **{pos}** sebagai kelas positif. Threshold = {threshold:.2f}.")
    a,b,c,d,e = st.columns(5)
    a.metric("Accuracy", format_percent(score_accuracy))
    b.metric("Precision", format_percent(score_precision))
    c.metric("Recall", format_percent(score_recall))
    d.metric("F1-score", f"{score_f1:.3f}")
    e.metric("ROC-AUC", f"{auc:.3f}")
    st.markdown("""
- **Accuracy:** dari seluruh prediksi testing, berapa proporsi yang benar?
- **Precision:** dari yang diprediksi sebagai kelas positif, berapa proporsi yang benar?
- **Recall:** dari seluruh mahasiswa yang sebenarnya kelas positif, berapa yang berhasil ditemukan?
- **F1-score:** ringkasan gabungan precision dan recall.
- **ROC-AUC:** kemampuan model membedakan kedua kelas pada beragam ambang, bukan pada threshold tunggal.
    """)
    st.info(f"**Pembanding (baseline sederhana):** jika kita selalu menebak kelas mayoritas pada **data training** ({pos if base_class == 1 else neg}), accuracy pada data testing = **{format_percent(baseline_accuracy)}**. Akurasi model perlu dibandingkan dengan baseline ini, bukan dinilai terpisah.")
    left, right = st.columns(2)
    with left:
        fig = plot_confusion(cm, neg, pos)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN={tn} · FP={fp} · FN={fn} · TP={tp}")
        st.write(f"**FN:** {fn} mahasiswa yang sebenarnya **{pos}**, tetapi diprediksi **{neg}**.")
    with right:
        if y_test.nunique() == 2:
            fig = plot_roc_curve(y_test, prob)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        st.caption("Kurva ROC dan ROC-AUC menggunakan probabilitas prediksi, sehingga tidak berubah hanya karena threshold digeser.")
    with st.expander("Koefisien model: untuk latihan interpretasi"):
        names = model.named_steps["persiapan"].get_feature_names_out()
        values = model.named_steps["regresi_logistik"].coef_[0]
        coef_df = pd.DataFrame({"Fitur setelah pemrosesan": names,
                                "Koefisien log-odds": values,
                                "Odds ratio (e^koefisien)": np.exp(values)}).sort_values("Koefisien log-odds", ascending=False)
        st.dataframe(coef_df.round(4), hide_index=True, use_container_width=True)
        st.caption("Koefisien numerik mengacu pada kenaikan satu standar deviasi; variabel kategori dibandingkan dengan kategori referensi. Besar koefisien bukan bukti sebab-akibat.")
    st.markdown("#### Unduh hasil prediksi data testing")
    output = df.loc[X_test.index].copy()
    output["Kelas_Aktual"] = np.where(y_test.to_numpy()==1, pos, neg)
    output["Probabilitas_" + str(pos).replace(" ", "_")] = prob.round(5)
    output["Kelas_Prediksi"] = np.where(predicted==1, pos, neg)
    output["Prediksi_Benar"] = np.where(y_test.to_numpy()==predicted, "Ya", "Tidak")
    st.dataframe(output.head(20), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Unduh prediksi data testing (CSV)",
                       output.to_csv(index=False).encode("utf-8-sig"),
                       file_name="hasil_regresi_logistik_testing.csv", mime="text/csv")
    st.warning("Evaluasi dari satu train-test split merupakan ilustrasi awal, bukan bukti bahwa model akan bekerja sama baiknya pada kampus atau tahun berbeda. Jangan gunakan data sintetis ini untuk keputusan nyata terhadap mahasiswa.")

with tab_single:
    st.subheader("Eksperimen: masukkan satu profil mahasiswa")
    st.write("Ubah nilai variabel X untuk melihat bagaimana probabilitas prediksi model berubah. Prediksi ini bukan kepastian dan hanya valid sebagai simulasi pada data sintetis.")
    with st.form("form_prediksi_profil"):
        entry = {}
        left, right = st.columns(2)
        for i, col in enumerate(chosen):
            with (left if i % 2 == 0 else right):
                if col in numeric_cols:
                    series = pd.to_numeric(X_train[col], errors="coerce")
                    mid = float(series.median()) if series.notna().any() else 0.0
                    entry[col] = st.number_input(col, value=round(mid, 2), step=1.0,
                                                  help=GUIDE.get(col, "Isi nilai numerik."))
                else:
                    choices = sorted(X_train[col].dropna().astype(str).unique().tolist())
                    if choices:
                        entry[col] = st.selectbox(col, choices)
                    else:
                        st.error(f"Tidak ada kategori pelatihan untuk fitur {col}.")
                        st.stop()
        submitted = st.form_submit_button("🔎 Prediksi profil mahasiswa", type="primary")
    if submitted:
        profile = pd.DataFrame([entry], columns=chosen)
        p = float(model.predict_proba(profile)[0, 1])
        guess = pos if p >= threshold else neg
        a, b = st.columns(2)
        a.metric(f"Probabilitas {pos}", f"{p*100:.1f}%")
        b.metric("Kelas hasil prediksi", guess)
        st.progress(max(0.0, min(p, 1.0)), text=f"Probabilitas kelas positif: {p:.1%}")
        st.caption(f"Kelas '{pos}' diprediksi jika probabilitas ≥ {threshold:.2f}. Perubahan satu fitur menunjukkan perbedaan keluaran model, bukan dampak kausal terjamin pada dunia nyata.")

with tab_questions:
    st.subheader("Lembar kerja: pertanyaan investigasi")
    st.write("Kerjakan pertanyaan setelah menggunakan menu Data, Model, Evaluasi, dan Simulasi Profil. Tulis **angka hasil dan interpretasinya**. Beberapa jawaban berubah ketika pengaturan model diubah.")
    student = st.text_input("Nama mahasiswa (opsional)")
    answers = []
    for i, (category, question, hint) in enumerate(QUESTIONS, start=1):
        st.markdown(f"**{i}. [{category}] {question}**")
        with st.expander("Petunjuk fitur"):
            st.caption(hint)
        a = st.text_area(f"Jawaban pertanyaan {i}", key=f"answer_{i}", height=75,
                         placeholder="Hasil angka/grafik yang diperoleh dan interpretasi singkat...")
        answers.append(a)
    report = ["LEMBAR PRAKTIKUM REGRESI LOGISTIK", f"Mahasiswa: {student or '(belum diisi)'}",
              f"Jumlah data: {len(df)}", f"Target: {target}", f"Kelas positif: {pos}",
              f"Training: {train_percent}%", f"Threshold: {threshold:.2f}",
              f"Fitur: {', '.join(chosen)}", ""]
    for i, ((cat, question, hint), answer) in enumerate(zip(QUESTIONS, answers), 1):
        report.extend([f"{i}. [{cat}] {question}", "Jawaban: " + (answer.strip() or "(belum diisi)"), ""])
    st.download_button("⬇️ Unduh lembar jawaban (.txt)",
                       data="\n".join(report).encode("utf-8"),
                       file_name="jawaban_praktikum_regresi_logistik.txt", mime="text/plain")
    st.caption("Dosen dapat menilai proses analisis, ketepatan membaca metrik, kualitas eksperimen, dan argumentasi mahasiswa.")

st.divider()
st.caption("STATCAL ONLINE — Prototipe pendidikan regresi logistik | Library: Streamlit, pandas, scikit-learn, Matplotlib | Dataset bawaan: sintetis")

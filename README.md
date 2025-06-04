# Indonesian News Article Summarizer

Aplikasi Streamlit untuk merangkum artikel berita berbahasa Indonesia secara otomatis menggunakan dua model transformer: **mBART** dan **PEGASUS**.

---

## Fitur

- **Input URL** artikel berita dari situs berita Indonesia.
- **Tampilkan isi artikel** secara otomatis.
- **Pilih model summarization**: mBART atau PEGASUS.
- **Ringkas artikel** dengan satu klik.
- **Tampilan ringkasan** yang rapi dan mudah dibaca.

---

## Cara Menjalankan di Lokal

1. **Clone repository ini**
    ```sh
    git clone https://github.com/USERNAME/indonesian-news-article-summarizer.git
    cd indonesian-news-article-summarizer
    ```

2. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

3. **Jalankan aplikasi**
    ```sh
    streamlit run streamlit_app.py
    ```

---

## Cara Menggunakan

1. Masukkan URL artikel berita berbahasa Indonesia.
2. Klik **"Tampilkan Artikel"** untuk melihat isi lengkap artikel.
3. Pilih model summarization (mBART atau PEGASUS) di dropdown.
4. Klik **"Ringkas"** untuk mendapatkan hasil ringkasan otomatis.

---

## Penjelasan Model

- **mBART**: Model multilingual yang sudah fine-tuned untuk Bahasa Indonesia, dapat langsung merangkum artikel berbahasa Indonesia.
- **PEGASUS**: Model fine-tuned pada dataset Bahasa Inggris. Artikel akan diterjemahkan ke Bahasa Inggris sebelum dirangkum, lalu hasil ringkasan diterjemahkan kembali ke Bahasa Indonesia.

---

## Deployment

Aplikasi ini dapat langsung dideploy ke [Streamlit Community Cloud](https://streamlit.io/cloud) dengan menghubungkan repository ini.

---

## Lisensi

MIT License

---

import streamlit as st
import torch
import re
from newspaper import Article
from langdetect import detect, DetectorFactory
from langcodes import Language

from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)
from googletrans import Translator

DetectorFactory.seed = 0

st.set_page_config(page_title="Indonesian News Summarizer", layout="wide")

# --- Custom Styles ---
st.markdown(
    """
    <style>
    .scroll-box {
        background-color: rgba(255, 255, 255, 0.95);
        color: #111;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ccc;
        max-height: 300px;
        overflow-y: scroll;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        position: relative;
    }
    .scroll-box::-webkit-scrollbar {
        width: 10px;
    }
    .scroll-box::-webkit-scrollbar-track {
        background: #e0e0e0;
        border-radius: 10px;
    }
    .scroll-box::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    .summary-box {
        background-color: rgba(227, 242, 253, 0.9);
        color: #0d0d0d;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #64b5f6;
        font-size: 1rem;
        font-weight: 500;
        line-height: 1.6;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .header-container {
        background-color: #0d47a1;
        color: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .usage-steps {
        padding: 1rem 2rem;
        background-color: #f5f5f5;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: #111;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    """
<div class="header-container">
    <h1>Indonesian News Summarizer</h1>
    <p>Ringkas artikel berita berbahasa Indonesia secara otomatis berdasarkan URL yang Anda masukkan.</p>
</div>
""",
    unsafe_allow_html=True,
)

# --- Step-by-step Instructions ---
st.markdown(
    """
<div class="usage-steps">
    <strong>Cara Menggunakan Aplikasi:</strong><br>
    1. Masukkan URL artikel berita dari situs berita berbahasa Indonesia.<br>
    2. Pilih model summarization yang ingin digunakan.<br>
    3. Klik tombol <b>Tampilkan Artikel</b> untuk melihat isi lengkap artikel.<br>
    4. Klik tombol <b>Ringkas</b> untuk mendapatkan hasil ringkasan otomatis.<br>
</div>
""",
    unsafe_allow_html=True,
)

# --- URL Input and Submit ---
st.markdown("### Masukkan URL Berita")
with st.form(key="url_form"):
    url = st.text_input(
        "",
        placeholder="https://www.cnnindonesia.com/...",
        label_visibility="collapsed",
    )
    submit_url = st.form_submit_button("Gunakan URL")

valid_url = re.match(r"https?://[\w\.-]+(?:/[\w\.-]*)*", url or "")

# --- Choose Model ---
model_option = st.selectbox(
    "Pilih Model Summarization:",
    ("Pilih model...", "mBART-large-50", "PEGASUS-large"),
    index=0,
    help="Pilih model yang ingin digunakan untuk merangkum artikel.",
    key="model_select",
    disabled=False,
    label_visibility="visible",
)

if submit_url:
    if not url or not valid_url:
        st.error(
            "Format URL tidak valid. Harap masukkan URL artikel berita yang benar."
        )
    else:
        st.markdown(
            "<p style='color:#66bb6a; font-size: 0.9rem;'>URL berhasil dimasukkan. Pilih model, lalu klik tombol di bawah untuk menampilkan artikel dan menghasilkan ringkasan.</p>",
            unsafe_allow_html=True,
        )

col1, col2 = st.columns(2)
with col1:
    show_btn = st.button("Tampilkan Artikel", use_container_width=True)
with col2:
    summarize_btn = st.button("Ringkas", use_container_width=True)


# --- Load Model and Tokenizer ---
@st.cache_resource
def load_mbart():
    model = MBartForConditionalGeneration.from_pretrained(
        "skripsi-summarization-1234/mbart-large-50-finetuned-xlsum-summarization"
    )
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "skripsi-summarization-1234/mbart-large-50-finetuned-xlsum-summarization"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device


@st.cache_resource
def load_pegasus():
    model = PegasusForConditionalGeneration.from_pretrained(
        "skripsi-summarization-1234/pegasus-large-finetuned-xlsum-summarization"
    )
    tokenizer = PegasusTokenizer.from_pretrained(
        "skripsi-summarization-1234/pegasus-large-finetuned-xlsum-summarization"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    translator = Translator()
    return model, tokenizer, device, translator


if model_option == "Pilih model...":
    st.info(
        "Silakan pilih model sebelum menampilkan artikel dan meringkas."
    )

# --- Show Article ---
if show_btn:
    if not url or not valid_url:
        st.error(
            "Format URL tidak valid. Harap masukkan URL artikel berita yang benar."
        )
    elif model_option == "Pilih model...":
        st.warning("Silakan pilih model terlebih dahulu.")
    else:
        with st.spinner("Mengambil artikel dan mendeteksi bahasa..."):
            try:
                article = Article(url, language="id")
                article.download()
                article.parse()
                lang = detect(article.text)
                if lang != "id":
                    lang_name = Language.get(lang).display_name("id")
                    st.error(
                        f"Artikel ini terdeteksi dalam bahasa {lang_name}. Aplikasi hanya mendukung ringkasan untuk berita Bahasa Indonesia."
                    )
                    st.session_state.article_text = None
                else:
                    st.session_state.article_text = article.text
            except Exception as e:
                st.error("Tidak dapat memuat artikel. Pastikan URL valid.")

# --- Re-render Article ---
if "article_text" in st.session_state and st.session_state.article_text:
    st.markdown("### Artikel Lengkap")
    st.markdown(
        f"<div class='scroll-box'>{st.session_state.article_text.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True,
    )

# --- Summarize Article ---
if summarize_btn:
    if model_option == "Pilih model...":
        st.warning("Silakan pilih model terlebih dahulu.")
    elif "article_text" in st.session_state and st.session_state.article_text:
        st.markdown(
            """
            <script>
            window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
            </script>
            """,
            unsafe_allow_html=True,
        )
        try:
            with st.spinner("Memproses artikel..."):
                if model_option == "mBART-large-50":
                    model, tokenizer, device = load_mbart()
                    tokenizer.src_lang = "id_ID"
                    inputs = tokenizer(
                        st.session_state.article_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding="longest",
                    ).to(device)
                    summary_ids = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True,
                        forced_bos_token_id=tokenizer.lang_code_to_id["id_ID"],
                    )
                    id_summary = tokenizer.decode(
                        summary_ids[0], skip_special_tokens=True
                    )
                elif model_option == "PEGASUS-large":
                    model, tokenizer, device, translator = load_pegasus()
                    en_text = translator.translate(
                        st.session_state.article_text, src="id", dest="en"
                    ).text
                    inputs = tokenizer(
                        en_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding="longest",
                    ).to(device)
                    summary_ids = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True,
                    )
                    en_summary = tokenizer.decode(
                        summary_ids[0], skip_special_tokens=True
                    )
                    id_summary = translator.translate(
                        en_summary, src="en", dest="id"
                    ).text

            st.success("Ringkasan berhasil dibuat!")
            st.markdown("### Hasil Ringkasan")
            st.markdown(
                f"<div class='summary-box'>{id_summary}</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat merangkum: {str(e)}")
    else:
        st.warning("Silakan tampilkan artikel terlebih dahulu.")

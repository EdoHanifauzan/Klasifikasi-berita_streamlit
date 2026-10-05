import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModel
from model import BiLSTMClassifier

# =====================================================
# Konfigurasi
# =====================================================

st.set_page_config(
    page_title="Klasifikasi Multi-Label Berita",
    page_icon="📰",
    layout="centered"
)

MODEL_NAME =  "indolem/indobert-base-uncased"

LABELS = [
    "Politik",
    "Hukum",
    "Pendidikan",
    "Olahraga",
    "Ekonomi"
]

THRESHOLD = 0.4

# =====================================================
# Load IndoBERT (hanya sekali)
# =====================================================

@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, bert_model = load_bert()

# =====================================================
# Fungsi Embedding
# =====================================================

def get_embedding(text, max_len):

    enc = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = bert_model(**enc)

    embedding = output.last_hidden_state[:, 0, :]

    return embedding

# =====================================================
# Load Model BiLSTM
# =====================================================

@st.cache_resource
def load_model(model_path):

    model = BiLSTMClassifier()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    return model

# =====================================================
# Tampilan Streamlit
# =====================================================

st.title("📰 Klasifikasi Multi-Label Judul Berita")

st.write(
    "Masukkan judul berita untuk mengetahui kategori yang diprediksi."
)

max_len = st.selectbox(
    "Pilih Max Length",
    [64, 128]
)

judul = st.text_input(
    "Masukkan Judul Berita"
)

# =====================================================
# Prediksi
# =====================================================

if st.button("Prediksi"):

    if judul.strip() == "":
        st.warning("Masukkan judul berita terlebih dahulu.")
        st.stop()

    if max_len == 64:
        model_path = "model6403.pth"
    else:
        model_path = "model03.pth"

    model = load_model(model_path)

    embedding = get_embedding(
        judul,
        max_len
    )

    embedding = embedding.unsqueeze(1)

    with torch.no_grad():
        pred = model(embedding)

    pred = pred.numpy()[0]

    # -----------------------------------------

    #st.subheader("Probabilitas Setiap Label")

    #for label, prob in zip(LABELS, pred):
    #    st.write(f"**{label}** : {prob:.4f}")

    # -----------------------------------------

    hasil = []

    for label, prob in zip(LABELS, pred):
        if prob >= THRESHOLD:
            hasil.append(label)

    st.subheader("Kategori Berita")

    if len(hasil) == 0:
        st.warning("Tidak ada kategori yang memenuhi threshold.")
    else:
        for h in hasil:
            st.success(h)

    # -----------------------------------------

    st.subheader("Ringkasan Probabilitas")

    cols = st.columns(len(LABELS))

    for col, label, prob in zip(cols, LABELS, pred):
        with col:
            st.metric(
                label=label,
                value=f"{prob:.2%}"
            )
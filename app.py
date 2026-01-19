import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =============================
# Config
# =============================
st.set_page_config(
    page_title="Perbandingan Model Penyakit Daun Pisang",
    layout="centered"
)

CLASS_NAMES = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']

# =============================
# Load models (Fixed Feature)
# =============================
@st.cache_resource
def load_models():
    model_mobilenet = tf.keras.models.load_model(
        "mobilenetv2_fixedfeature.keras", compile=False
    )
    model_efficientnet = tf.keras.models.load_model(
        "efficientnetb0_fixedfeature.keras", compile=False
    )
    return model_mobilenet, model_efficientnet

model_mn, model_ef = load_models()

# =============================
# Preprocessing
# SESUAI KODE SKRIPSI:
# - resize 224x224
# - TANPA /255
# - preprocess_input ADA DI DALAM MODEL
# =============================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype="float32")
    img = np.expand_dims(img, axis=0)
    return img

# =============================
# UI
# =============================
st.title("Perbandingan MobileNetV2 vs EfficientNetB0 (Fixed Feature)")
st.write(
    "Aplikasi ini membandingkan hasil klasifikasi penyakit daun pisang "
    "menggunakan **MobileNetV2 dan EfficientNetB0 pada skenario Fixed Feature (FE)** "
    "dengan citra input yang sama."
)

uploaded_file = st.file_uploader(
    "Upload gambar daun pisang",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # ===== TAMPILAN GAMBAR DIPERKECIL & RAPIH =====
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(
            image,
            caption="Gambar Input",
            width=380   # ⬅️ INI YANG MEMPERKECIL GAMBAR
        )

    x = preprocess_image(image)

    # =============================
    # Prediction
    # =============================
    pred_mn = model_mn.predict(x, verbose=0)[0]
    pred_ef = model_ef.predict(x, verbose=0)[0]

    idx_mn = int(np.argmax(pred_mn))
    idx_ef = int(np.argmax(pred_ef))

    with col_result:
        st.subheader("Hasil Prediksi")

        st.markdown("### MobileNetV2 (Fixed Feature)")
        st.write(f"**Prediksi:** {CLASS_NAMES[idx_mn]}")
        st.write(f"**Confidence:** {pred_mn[idx_mn]*100:.2f}%")

        st.markdown("### EfficientNetB0 (Fixed Feature)")
        st.write(f"**Prediksi:** {CLASS_NAMES[idx_ef]}")
        st.write(f"**Confidence:** {pred_ef[idx_ef]*100:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Probabilitas MobileNetV2")
        st.bar_chart(
            {CLASS_NAMES[i]: float(pred_mn[i]) for i in range(len(CLASS_NAMES))}
        )

    with col2:
        st.subheader("Distribusi Probabilitas EfficientNetB0")
        st.bar_chart(
            {CLASS_NAMES[i]: float(pred_ef[i]) for i in range(len(CLASS_NAMES))}
        )

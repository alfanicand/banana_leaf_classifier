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
CONF_THRESHOLD = 0.50  # 70%

# =============================
# Load all models
# =============================
@st.cache_resource
def load_models():
    models = {
        "Fixed Feature": {
            "MobileNetV2": tf.keras.models.load_model("mobilenetv2_fixedfeature.keras", compile=False),
            "EfficientNetB0": tf.keras.models.load_model("efficientnetb0_fixedfeature.keras", compile=False)
        },
        "FT10": {
            "MobileNetV2": tf.keras.models.load_model("mobilenetv2_ft10.keras", compile=False),
            "EfficientNetB0": tf.keras.models.load_model("efficientnetb0_ft10.keras", compile=False)
        },
        "FT20": {
            "MobileNetV2": tf.keras.models.load_model("mobilenetv2_ft20.keras", compile=False),
            "EfficientNetB0": tf.keras.models.load_model("efficientnetb0_ft20.keras", compile=False)
        },
        "FT30": {
            "MobileNetV2": tf.keras.models.load_model("mobilenetv2_ft30.keras", compile=False),
            "EfficientNetB0": tf.keras.models.load_model("efficientnetb0_ft30.keras", compile=False)
        }
    }
    return models

MODELS = load_models()

# =============================
# Preprocessing (SESUAI SKRIPSI)
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
st.title("Klasifikasi Penyakit Daun Pisang")
st.write(
    "Sistem ini membandingkan hasil klasifikasi penyakit daun pisang "
    "menggunakan MobileNetV2 dan EfficientNetB0 pada berbagai skenario pelatihan."
)

# ===== Model selector =====
variant = st.selectbox(
    "Pilih skenario model",
    ["Fixed Feature", "FT10", "FT20", "FT30"]
)

uploaded_file = st.file_uploader(
    "Upload gambar daun pisang",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Gambar Input")
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image(image, use_container_width=True)

    x = preprocess_image(image)

    # =============================
    # Prediction
    # =============================
    model_mn = MODELS[variant]["MobileNetV2"]
    model_ef = MODELS[variant]["EfficientNetB0"]

    pred_mn = model_mn.predict(x, verbose=0)[0]
    pred_ef = model_ef.predict(x, verbose=0)[0]

    conf_mn = float(np.max(pred_mn))
    conf_ef = float(np.max(pred_ef))

    idx_mn = int(np.argmax(pred_mn))
    idx_ef = int(np.argmax(pred_ef))

    # =============================
    # CONFIDENCE GATE
    # =============================
    if conf_mn < CONF_THRESHOLD or conf_ef < CONF_THRESHOLD:
        st.markdown("---")
        st.markdown(
            "<h4 style='text-align:center; color:red;'>"
            "Silakan upload ulang gambar daun pisang"
            "</h4>",
            unsafe_allow_html=True
        )
        st.caption(
            "Gambar mungkin bukan daun pisang"
        )
    else:
        st.markdown("---")
        st.subheader(f"Hasil Prediksi ({variant})")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### MobileNetV2")
            st.write(f"**Prediksi:** {CLASS_NAMES[idx_mn]}")
            st.write(f"**Confidence:** {conf_mn*100:.2f}%")
            st.bar_chart(
                {CLASS_NAMES[i]: float(pred_mn[i]) for i in range(len(CLASS_NAMES))}
            )

        with col2:
            st.markdown("### EfficientNetB0")
            st.write(f"**Prediksi:** {CLASS_NAMES[idx_ef]}")
            st.write(f"**Confidence:** {conf_ef*100:.2f}%")
            st.bar_chart(
                {CLASS_NAMES[i]: float(pred_ef[i]) for i in range(len(CLASS_NAMES))}
            )



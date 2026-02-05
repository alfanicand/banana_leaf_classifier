import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =============================
# Config
# =============================
st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Pisang",
    layout="centered"
)

CLASS_NAMES = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']
CONF_THRESHOLD = 0.60   # threshold confidence (60%)

# =============================
# Load models
# =============================
@st.cache_resource
def load_models():
    models = {
        "Fixed Feature": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_fixedfeature.keras", compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_fixedfeature.keras", compile=False
            ),
        },
        "Fine-Tuning FT10": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_ft10.keras", compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_ft10.keras", compile=False
            ),
        },
        "Fine-Tuning FT20": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_ft20.keras", compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_ft20.keras", compile=False
            ),
        },
        "Fine-Tuning FT30": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_ft30.keras", compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_ft30.keras", compile=False
            ),
        },
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
    "Aplikasi ini melakukan klasifikasi penyakit daun pisang menggunakan "
    "**MobileNetV2 dan EfficientNetB0** dengan skenario **Fixed Feature** dan **Fine-Tuning**."
)

# ===== Dropdown skenario =====
scenario = st.selectbox(
    "Pilih skenario model",
    options=list(MODELS.keys())
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
        st.image(image, caption="Citra input", use_container_width=True)

    x = preprocess_image(image)

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns(2)

    # =============================
    # MobileNetV2
    # =============================
    with col1:
        st.markdown(f"### MobileNetV2 ({scenario})")
        model_mn = MODELS[scenario]["MobileNetV2"]
        pred_mn = model_mn.predict(x, verbose=0)[0]

        idx_mn = int(np.argmax(pred_mn))
        conf_mn = float(pred_mn[idx_mn])

        st.write(f"**Prediksi:** {CLASS_NAMES[idx_mn]}")
        st.write(f"**Confidence:** {conf_mn*100:.2f}%")

        if conf_mn < CONF_THRESHOLD:
            st.warning(
                "⚠️ Confidence rendah. "
                "Gambar kemungkinan **bukan daun pisang** atau kualitas citra kurang baik."
            )

        st.bar_chart(
            {CLASS_NAMES[i]: float(pred_mn[i]) for i in range(len(CLASS_NAMES))}
        )

    # =============================
    # EfficientNetB0
    # =============================
    with col2:
        st.markdown(f"### EfficientNetB0 ({scenario})")
        model_ef = MODELS[scenario]["EfficientNetB0"]
        pred_ef = model_ef.predict(x, verbose=0)[0]

        idx_ef = int(np.argmax(pred_ef))
        conf_ef = float(pred_ef[idx_ef])

        st.write(f"**Prediksi:** {CLASS_NAMES[idx_ef]}")
        st.write(f"**Confidence:** {conf_ef*100:.2f}%")

        if conf_ef < CONF_THRESHOLD:
            st.warning(
                "⚠️ Confidence rendah. "
                "Gambar kemungkinan **bukan daun pisang** atau kualitas citra kurang baik."
            )

        st.bar_chart(
            {CLASS_NAMES[i]: float(pred_ef[i]) for i in range(len(CLASS_NAMES))}
        )

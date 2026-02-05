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
CONF_THRESHOLD = 0.60  # 60%

# =============================
# Load models
# =============================
@st.cache_resource
def load_models():
    return {
        "Fixed Feature": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_fixedfeature.keras", compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_fixedfeature.keras", compile=False
            ),
        },
        "FT10": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_ft10.keras", compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_ft10.keras", compile=False
            ),
        },
        "FT20": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_ft20.keras", compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_ft20.keras", compile=False
            ),
        },
        "FT30": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_ft30.keras", compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_ft30.keras", compile=False
            ),
        },
    }

MODELS = load_models()

# =============================
# Preprocessing (sesuai skripsi)
# =============================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype="float32")
    img = np.expand_dims(img, axis=0)
    return img

def validate_prediction(pred, threshold):
    max_conf = float(np.max(pred))
    return max_conf >= threshold, max_conf

# =============================
# UI
# =============================
st.title("Klasifikasi Penyakit Daun Pisang")
st.write(
    "Aplikasi ini menampilkan hasil klasifikasi penyakit daun pisang "
    "menggunakan **MobileNetV2 dan EfficientNetB0** "
    "dengan skenario **Fixed Feature dan Fine-Tuning**."
)

scenario = st.selectbox(
    "Pilih skenario model",
    ["Fixed Feature", "FT10", "FT20", "FT30"]
)

uploaded_file = st.file_uploader(
    "Upload gambar daun pisang",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # ===== TAMPILAN GAMBAR DI TENGAH =====
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

        valid_mn, conf_mn = validate_prediction(pred_mn, CONF_THRESHOLD)

        if not valid_mn:
            st.warning(
                "⚠️ **Confidence prediksi rendah.**\n\n"
                "Gambar kemungkinan **bukan daun pisang** "
                "atau objek tidak terlihat jelas.\n\n"
                "**Silakan upload ulang gambar daun pisang.**"
            )
        else:
            idx = int(np.argmax(pred_mn))
            st.write(f"**Prediksi:** {CLASS_NAMES[idx]}")
            st.write(f"**Confidence:** {conf_mn*100:.2f}%")

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

        valid_ef, conf_ef = validate_prediction(pred_ef, CONF_THRESHOLD)

        if not valid_ef:
            st.warning(
                "⚠️ **Confidence prediksi rendah.**\n\n"
                "Gambar kemungkinan **bukan daun pisang** "
                "atau kualitas citra tidak sesuai.\n\n"
                "**Silakan upload ulang gambar daun pisang.**"
            )
        else:
            idx = int(np.argmax(pred_ef))
            st.write(f"**Prediksi:** {CLASS_NAMES[idx]}")
            st.write(f"**Confidence:** {conf_ef*100:.2f}%")

            st.bar_chart(
                {CLASS_NAMES[i]: float(pred_ef[i]) for i in range(len(CLASS_NAMES))}
            )

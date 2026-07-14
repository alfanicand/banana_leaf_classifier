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

CONF_THRESHOLDS = {
    "Fixed Feature": 0.50,
    "FT10": 0.53,
    "FT20": 0.51,
    "FT30": 0.50
}


# =============================
# Load all models
# =============================
@st.cache_resource
def load_model(variant):

    model_paths = {
        "Fixed Feature": {
            "MobileNetV2": "mobilenetv2_fixedfeature.keras",
            "EfficientNetB0": "efficientnetb0_fixedfeature.keras"
        },
        "FT10": {
            "MobileNetV2": "mobilenetv2_ft10.keras",
            "EfficientNetB0": "efficientnetb0_ft10.keras"
        },
        "FT20": {
            "MobileNetV2": "mobilenetv2_ft20.keras",
            "EfficientNetB0": "efficientnetb0_ft20.keras"
        },
        "FT30": {
            "MobileNetV2": "mobilenetv2_ft30.keras",
            "EfficientNetB0": "efficientnetb0_ft30.keras"
        }
    }


    models = {
        "MobileNetV2": tf.keras.models.load_model(
            model_paths[variant]["MobileNetV2"],
            compile=False
        ),

        "EfficientNetB0": tf.keras.models.load_model(
            model_paths[variant]["EfficientNetB0"],
            compile=False
        )
    }

    return models

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
st.markdown("---")
st.subheader("Informasi Kelas Penyakit")

with st.expander("🟤 Cordana"):
    st.write("Bercak coklat hingga keabu-abuan pada permukaan daun.")
    st.write("Penanganan: Mengatur kelembaban kebun dengan jarak tanam yang tidak terlalu rapat dan menghindari naungan berlebih, serta penggunaan fungisida berbahan aktif mankozeb atau propineb setelah pengamatan rutin.")

with st.expander("🟢 Healthy"):
    st.write("Daun dalam kondisi sehat tanpa gejala penyakit.")
    st.write("Penanganan: Pemeliharaan rutin melalui pemupukan seimbang, pengaturan jarak tanam, serta pemantauan berkala untuk menjaga kondisi tanaman tetap sehat.")

with st.expander("🟠 Pestalotiopsis"):
    st.write("Bercak tidak beraturan berwarna coklat dengan tepi lebih gelap.")
    st.write("Penanganan: Pemangkasan dan pemusnahan daun terinfeksi untuk mencegah penyebaran spora, aplikasi fungisida berbahan aktif difenokonazol atau azoksistrobin, serta pemupukan tinggi kalium dan pengaturan jarak tanam untuk mengurangi kelembapan.")

with st.expander("🟡 Sigatoka"):
    st.write("Bercak kecil memanjang berwarna kuning hingga coklat.")
    st.write("Penanganan: Pemupukan untuk menjaga kesuburan tanah, pemusnahan daun terinfeksi, dan aplikasi fungisida berbahan aktif mankozeb atau propineb.")

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
    MODELS = load_model(variant)

    model_mn = MODELS["MobileNetV2"]
    model_ef = MODELS["EfficientNetB0"]

    pred_mn = model_mn.predict(x, verbose=0)[0]
    pred_ef = model_ef.predict(x, verbose=0)[0]

    conf_mn = float(np.max(pred_mn))
    conf_ef = float(np.max(pred_ef))

    idx_mn = int(np.argmax(pred_mn))
    idx_ef = int(np.argmax(pred_ef))

    threshold = CONF_THRESHOLDS[variant]
    
    # =============================
    # CONFIDENCE GATE
    # =============================
    if conf_mn < threshold or conf_ef < threshold:
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
















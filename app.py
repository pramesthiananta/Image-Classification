import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(
    page_title="Klasifikasi Buah",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 Klasifikasi Buah (Apple vs Mango)")
st.write("Upload model dan gambar untuk melakukan klasifikasi buah.")

# Threshold untuk menolak gambar yang tidak dikenali
THRESHOLD = 0.80

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# ==========================
# Upload Model
# ==========================
st.subheader("1️⃣ Upload Model")

uploaded_model = st.file_uploader(
    "Pilih file model (.keras)",
    type=["keras"]
)

if uploaded_model is not None:

    temp_model_path = "model_buah.keras"

    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())

    try:
        model = load_model(temp_model_path)

        st.success("✅ Model berhasil dimuat")

        # ==========================
        # Upload Gambar Uji
        # ==========================
        st.subheader("2️⃣ Upload Gambar Uji")

        uploaded_image = st.file_uploader(
            "Pilih gambar buah (.jpg/.jpeg/.png)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:

            img = Image.open(uploaded_image).convert("RGB")

            st.image(
                img,
                caption="Gambar Uji",
                use_container_width=True
            )

            # Ubah ukuran sesuai model
            img_resized = img.resize((227, 227))

            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            hasil = model.predict(img_array, verbose=0)

            class_names = ["apple", "mango"]

            prediksi = np.argmax(hasil)
            confidence = np.max(hasil)

            st.subheader("📊 Hasil Prediksi")

            # Jika confidence rendah
            if confidence < THRESHOLD:
                st.warning("⚠️ Saya tidak tahu")
            else:
                st.success(
                    f"✅ Buah terdeteksi sebagai: **{class_names[prediksi].upper()}**"
                )

            st.write(
                f"**Tingkat Keyakinan:** {confidence * 100:.2f}%"
            )

            st.subheader("Probabilitas Tiap Kelas")

            for i, nama in enumerate(class_names):
                st.write(
                    f"**{nama.capitalize()}:** {hasil[0][i] * 100:.2f}%"
                )

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")

else:
    st.info("Silakan upload file model .keras terlebih dahulu.")

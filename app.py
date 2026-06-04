
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(page_title="Klasifikasi Buah", page_icon="🍎")

st.title("🍎 Klasifikasi Buah (Apple vs Mango)")

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

uploaded_model = st.file_uploader(
    "Upload model (.keras)",
    type=["keras"]
)

if uploaded_model is not None:
    temp_model_path = "model_buah.keras"

    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())

    try:
        model = load_model(temp_model_path)
        st.success("Model berhasil dimuat")

        class_names = ["apple", "mango"]

        uploaded_image = st.file_uploader(
            "Upload gambar buah",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            img = Image.open(uploaded_image).convert("RGB")

            st.image(img, caption="Gambar Uji", use_container_width=True)

            img_resized = img.resize((227, 227))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            hasil = model.predict(img_array, verbose=0)
            prediksi = np.argmax(hasil)

            st.subheader("Hasil Prediksi")
            st.write(f"**Label:** {class_names[prediksi]}")
            st.write("**Probabilitas:**")
            st.write(hasil)

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
else:
    st.info("Silakan upload file model .keras terlebih dahulu.")

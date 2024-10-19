import joblib
import streamlit as st
import numpy as np
from PIL import Image
import os

# Dosya yolunu tanımlayın
file_path = os.path.join(os.getcwd(), "mnist_veri_seti.joblib")

# Modeli yükleyin
if os.path.exists(file_path):
    loaded_model = joblib.load(file_path)
    st.write("Model başarıyla yüklendi.")
else:
    st.write(f"Hata: {file_path} dosyası bulunamadı.")
    st.stop()  # Dosya bulunamazsa uygulamayı durdurun

st.title("Rakam Tahmin Etme Uygulaması")

# Resim yükleme
uploaded_file = st.file_uploader("Bir resim yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((28, 28))  # Resmi 28x28 boyutuna yeniden boyutlandır
    img = img.convert("L")  # Grayscale (siyah-beyaz) formata çevir
    img_array = np.array(img).reshape(1, -1)  # Resmi düzleştir

    # Tahmin yap
    prediction = loaded_model.predict(img_array)

    # Sonucu göster
    st.write(f"Tahmin Edilen Sınıf: {prediction[0]}")
    st.image(img, caption=f"Tahmin Edilen Sınıf: {prediction[0]}")

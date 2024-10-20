import streamlit as st
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Başlık
st.title("Rakam Tanıma Uygulaması")

# Dosya yolunu belirle
model_path = os.path.join(os.path.dirname(__file__), 'mnist_veri_seti.joblib')

# Modeli yükle
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Kullanıcıdan resim yüklemesi iste
uploaded_file = st.file_uploader("Lütfen bir resim dosyası yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Resmi göster
    img = Image.open(uploaded_file)
    st.image(img, caption='Yüklenen Resim', use_column_width=True)

    # Resmi işleme
    img = img.resize((28, 28))
    img = img.convert("L")
    img_array = np.array(img).reshape(1, -1)

    # Tahmin yap
    pred = loaded_model.predict(img_array)
    st.write(f"Tahmin edilen rakam: {pred[0]}")
else:
    st.write("Lütfen bir resim yükleyin.")

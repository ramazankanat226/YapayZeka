import os
import joblib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Dinamik olarak modelin dosya yolunu belirleyin
model_path = os.path.join(os.path.dirname(__file__), 'mnist_veri_seti_streamlit.joblib')

# Modeli tekrar yükleme
lr_loaded = joblib.load(model_path)
st.write("Model yüklendi.")

# Streamlit uygulama başlığı
st.title("MNIST Rakam Tanıma Uygulaması")

# Kullanıcıdan resim yüklemesini isteyin
uploaded_file = st.file_uploader("Lütfen bir resim yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Resmi açma ve işleme
    img = Image.open(uploaded_file)
    img = img.resize((28, 28))
    img = img.convert("L")
    
    # Resmi numpy dizisine çevirme ve modelin kabul edeceği boyuta getirme
    img_array = np.array(img).reshape(1, -1)
    
    # Modelle tahmin yapma
    pred = lr_loaded.predict(img_array)
    st.write(f"Tahmin edilen sınıf: {pred[0]}")
    
    # Tahmin edilen rakamı görselleştirme
    st.image(img, caption=f"Tahmin edilen rakam: {pred[0]}")

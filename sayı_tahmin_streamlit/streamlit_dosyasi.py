import joblib
import streamlit as st
import numpy as np
from PIL import Image

st.title("rakam tahmin etme uygulaması")

loaded_model=joblib.load("mnist_veri_seti.joblib")

uploaded_file=st.file_uploader("bir resim yükleyin",type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img=Image.open(uploaded_file)
    img=img.resize((28,28))
    img=img.convert("L")
    img_array=np.array(img).reshape(1,-1)
    
    prediction=loaded_model.predict(img_array)
    
    st.write(f"tahmin edilen sınıf {prediction[0]}")
    
    st.image(img,caption=f"tahmin edilen sınıf {prediction}")
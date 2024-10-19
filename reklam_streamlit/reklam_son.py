import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# GitHub'daki CSV dosyasının URL'si
url = 'https://raw.githubusercontent.com/ramazankanat226/YapayZeka/refs/heads/main/reklam_streamlit/reklam.csv'

# CSV dosyasını GitHub'dan okuma
data = pd.read_csv(url)

st.title("Reklam Harcamaları")
st.write("Veri Başlıkları")
st.write(data.head())  # Veri çerçevesinin ilk birkaç satırını görüntüle

# Özelliklerin ve Hedef Değerlerin Seçilmesi
x = data.iloc[:, 1:-1].values  # Tüm satırlar ve 1. sütundan son sütuna kadar değerler (bağımsız değişkenler)
y = data.iloc[:, -1].values  # Son sütundaki değerler (bağımlı değişken)

# Eğitim ve Test Verilerine Bölme
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=22)

# Modelin Tanımlanması ve Eğitilmesi
lr = LinearRegression()
lr.fit(xtrain, ytrain)

# Tahmin Yapılması
yhead = lr.predict(xtest)

# Kullanıcı Girdisi Alma
tv_input = st.number_input("TV için reklam harcama tutarı girin", min_value=0, value=44)
radio_input = st.number_input("Radyo için reklam harcama tutarı girin", min_value=0, value=48)
gazete_input = st.number_input("Gazete için reklam harcama tutarı girin", min_value=0, value=35)

# Tahmin İçin Verilerin Hazırlanması
value = np.array([[tv_input, radio_input, gazete_input]])
predict_value = lr.predict(value)

# Tahmin Sonucunun Görüntülenmesi
st.write("Verilen değerlerin tahmini:", predict_value[0])  # Tahmin edilen değeri göster

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st


warnings.filterwarnings("ignore")


data = pd.read_csv("magaza_yorumlari_duygu_analizi.csv", encoding="utf-16")
data = data.dropna()
data["Durum"] = data.loc[:, "Durum"].map({"Olumlu": 0, "Tarafsız": 1, "Olumsuz": 2})


with open("turkce_stopwords.txt", "r", encoding="utf-8") as file:
    stop_words = set(file.read().split())

def harfdegistir(cumle):
    cumle = re.sub("[^a-zA-ZğĞüÜşŞİıöÖçÇ ]", " ", cumle)  # Boşluk bırak
    cumle = cumle.lower()
    
    cumle = cumle.split()
    
    cumle = [word for word in cumle if word not in stop_words]
    cumle = " ".join(cumle)
    return cumle


x = data["Görüş"].values
songorus = []
for i in range(len(x)):
    x1 = harfdegistir(x[i])
    songorus.append(x1)


max_feature = 2500
cv = CountVectorizer(max_features=max_feature)
space_matrix = cv.fit_transform(songorus).toarray()


x = space_matrix
y = data.iloc[:, 1].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=33)


rf = RandomForestClassifier(n_estimators=100, random_state=45)
rf.fit(xtrain, ytrain)



# Streamlit uygulaması
st.title("Yorum İnceleme Uygulaması")
yorum_input = st.text_input("Yorumu girin")


accuracy = rf.score(xtest, ytest)
st.write(f"Model Doğruluğu: {accuracy:.2f}")

def giris_cumlesi_siniflandir(cumle):
    temiz_cumle = harfdegistir(cumle)
    if temiz_cumle:  
        vektor = cv.transform([temiz_cumle]).toarray()
        tahmin = rf.predict(vektor)
        if tahmin == 0:
            st.write("Girilen yorum olumlu")
        elif tahmin == 1:
            st.write("Girilen yorum tarafsız")
        else:
            st.write("Girilen yorum olumsuz")
    else:
        st.write("Girilen yorum sadece stop word içeriyor veya boş.")

giris_cumlesi_siniflandir(yorum_input)

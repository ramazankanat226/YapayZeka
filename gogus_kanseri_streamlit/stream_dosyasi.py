import streamlit as st
import pickle
import pandas as pd
import os


# Başlık
st.title("Göğüs Kanseri Teşhis Uygulaması")


# Model ve scaler dosyalarını yükle
with open("scaler1.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)

with open("lr1.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Kullanıcıdan manuel olarak özellik değerlerini al
st.header("Özellik Değerlerini Giriniz:")

def user_input_features():
    radius_mean = st.slider("Radius Mean", min_value=0.0, max_value=30.0, value=17.99)
    texture_mean = st.slider("Texture Mean", min_value=0.0, max_value=40.0, value=10.38)
    perimeter_mean = st.slider("Perimeter Mean", min_value=0.0, max_value=200.0, value=122.8)
    area_mean = st.slider("Area Mean", min_value=0.0, max_value=2500.0, value=1001.0)
    smoothness_mean = st.slider("Smoothness Mean", min_value=0.0, max_value=0.2, value=0.1184)
    compactness_mean = st.slider("Compactness Mean", min_value=0.0, max_value=0.5, value=0.2776)
    concavity_mean = st.slider("Concavity Mean", min_value=0.0, max_value=0.5, value=0.3001)
    concave_points_mean = st.slider("Concave Points Mean", min_value=0.0, max_value=0.2, value=0.1471)
    symmetry_mean = st.slider("Symmetry Mean", min_value=0.0, max_value=0.5, value=0.2419)
    fractal_dimension_mean = st.slider("Fractal Dimension Mean", min_value=0.0, max_value=0.1, value=0.07871)
    radius_se = st.slider("Radius SE", min_value=0.0, max_value=5.0, value=1.095)
    texture_se = st.slider("Texture SE", min_value=0.0, max_value=5.0, value=0.9053)
    perimeter_se = st.slider("Perimeter SE", min_value=0.0, max_value=20.0, value=8.589)
    area_se = st.slider("Area SE", min_value=0.0, max_value=200.0, value=153.4)
    smoothness_se = st.slider("Smoothness SE", min_value=0.0, max_value=0.01, value=0.006399)
    compactness_se = st.slider("Compactness SE", min_value=0.0, max_value=0.1, value=0.04904)
    concavity_se = st.slider("Concavity SE", min_value=0.0, max_value=0.1, value=0.05373)
    concave_points_se = st.slider("Concave Points SE", min_value=0.0, max_value=0.05, value=0.01587)
    symmetry_se = st.slider("Symmetry SE", min_value=0.0, max_value=0.1, value=0.03003)
    fractal_dimension_se = st.slider("Fractal Dimension SE", min_value=0.0, max_value=0.01, value=0.006193)
    radius_worst = st.slider("Radius Worst", min_value=0.0, max_value=40.0, value=25.38)
    texture_worst = st.slider("Texture Worst", min_value=0.0, max_value=50.0, value=17.33)
    perimeter_worst = st.slider("Perimeter Worst", min_value=0.0, max_value=250.0, value=184.6)
    area_worst = st.slider("Area Worst", min_value=0.0, max_value=3000.0, value=2019.0)
    smoothness_worst = st.slider("Smoothness Worst", min_value=0.0, max_value=0.3, value=0.1622)
    compactness_worst = st.slider("Compactness Worst", min_value=0.0, max_value=1.0, value=0.6656)
    concavity_worst = st.slider("Concavity Worst", min_value=0.0, max_value=1.0, value=0.7119)
    concave_points_worst = st.slider("Concave Points Worst", min_value=0.0, max_value=0.5, value=0.2654)
    symmetry_worst = st.slider("Symmetry Worst", min_value=0.0, max_value=0.7, value=0.4601)
    fractal_dimension_worst = st.slider("Fractal Dimension Worst", min_value=0.0, max_value=0.2, value=0.1189)
    
    data = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Özellikleri ölçeklendir
ozellikler_olceklendirilmis = loaded_scaler.transform(input_df)

# Tahmin yap
tahmin = loaded_model.predict(ozellikler_olceklendirilmis)

# Sonuçları göster
st.subheader('Tahmin Edilen Teşhis Durumu:')
st.write('Kötü Huylu' if tahmin[0] == 1 else 'İyi Huylu')
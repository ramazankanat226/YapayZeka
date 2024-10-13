import numpy as np
import pandas as pd

data=pd.read_csv("reklam.csv")
data.head()
x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=22)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(xtrain,ytrain)

yhead=lr.predict(xtest)
#%%
import streamlit as st

st.title("Reklam harcamaları")
st.write("Veri Başlıkları")
st.write(data.head())

tv_input=st.number_input("tv için reklam harcama tutarı girin",min_value=0,value=44)
radio_input=st.number_input("radio için reklam harcama tutarı girin",min_value=0,value=48)
gazete_input=st.number_input("gazete için reklam harcama tutarı girin",min_value=0,value=35)

value=np.array([[tv_input,radio_input,gazete_input]])

predict_value=lr.predict(value)

st.write("verilen değerlerin tahmini ",predict_value)
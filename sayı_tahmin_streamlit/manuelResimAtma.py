from sklearn import datasets
x, y = datasets.fetch_openml("mnist_784", version=1, return_X_y=True)
#%%
x.head()

#%%
y.head()
#%%
x.shape
#%%
y.shape
#%%
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    digit = x.iloc[i]  # İlk 9 rakamı seçiyoruz
    digit_pixel = np.array(digit).reshape(28, 28)  # Veriyi 28x28 boyutuna dönüştürüyoruz
    ax.imshow(digit_pixel, cmap='gray')  # Görüntüyü göster
    ax.set_title(f"örnek {i+1}")  # Başlık ekle
plt.tight_layout()
plt.show()

#%%

digit=x.iloc[16]
digit_pixel=np.array(digit).reshape(28,28)
#%%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=23)
#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=3000)  # Lojistik regresyon modelini tanımla
#%%
lr.fit(xtrain,ytrain)
#%%
from sklearn.metrics import accuracy_score
ypred = lr.predict(xtest)  # Test verisi ile tahmin yap
accuracy = accuracy_score(ytest, ypred)  # Doğruluk oranını hesapla
print("doğruluk oranı:", accuracy)
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)  # Karışıklık matrisini oluştur
print("confusion matrix \n", cm)
#%%
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")  # Karışıklık matrisini ısı haritası olarak göster
ax.set_xlabel("tahmin", fontsize=14)
ax.set_ylabel("gerçek değer", fontsize=14)
plt.show()
#%%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("5.png")  # Yeni görüntüyü yükle
img = img.resize((28, 28))  # Görüntüyü 28x28 boyutuna getir
img = img.convert("L")  # Görüntüyü gri tonlamaya çevir
img_array = np.array(img).reshape(1, -1)  # Görüntüyü düzleştir


#%%
pred=lr.predict(img_array)
#%%
print("tahmin edilen sayı:",pred)

#%%
import joblib
joblib.dump(lr, "mnist_veri_seti.joblib")  # Modeli joblib ile kayde





















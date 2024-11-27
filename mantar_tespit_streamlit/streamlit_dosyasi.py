import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import gdown

# Google Drive'dan dosya indirme
url = 'https://drive.google.com/uc?id=1_9Qm0Al1RjhNrtcNuJMAU0O4q1NEurb5' 
output = 'mantar_best.pt'  # İndirilen dosya ismi
gdown.download(url, output, quiet=False)

# Modeli yükle
model = YOLO("mantar_best.pt")  # Model dosyasının yolu
st.title("Kuzu Göbeği Mantarı Algılama Uygulaması")

# Seçenekler (menü)
option = st.selectbox("Bir seçenek seçin", ["Fotoğraf Yükle", "Video Yükle", "Kamera ile Canlı Video Çekimi ve Tespit"])

# Yardımcı Fonksiyonlar
def process_frame(frame, model):
    """Her kareyi işleyen fonksiyon"""
    results = model.predict(source=frame, conf=0.5)
    detections = results[0].boxes.data.cpu().numpy()
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame

def save_uploaded_file(uploaded_file, suffix):
    """Yüklenen dosyayı geçici dosyaya kaydeden fonksiyon"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

if option == "Fotoğraf Yükle":
    uploaded_image = st.file_uploader("Fotoğraf yükleyin", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.read())
            image_path = temp_file.name

        image = cv2.imread(image_path)

        # Modeli kullanarak tahmin yap
        results = model.predict(source=image, conf=0.5)
        detections = results[0].boxes.data.cpu().numpy()

        # Bounding box çizimi
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            if confidence >= 0.6:  # Doğruluk oranı %60 ve üzeriyse
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Kırmızı kutu
                cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # İşlenmiş görüntüyü gösterme
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="İşlenmiş Fotoğraf")

        # Fotoğrafı indirme
        _, img_encoded = cv2.imencode('.jpg', image)
        st.download_button(
            label="İndir",
            data=img_encoded.tobytes(),
            file_name="islenmis_resim.jpg",
            mime="image/jpeg"
        )

# Video Yükle
elif option == "Video Yükle":
    uploaded_video = st.file_uploader("Video yükleyin", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Geçici dosya oluşturma
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_video.read())
            video_path = temp_file.name

        # Video okuma
        cap = cv2.VideoCapture(video_path)

        # Çıktı videosunu kaydetmek için (geçici dosya)
        output_video_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec seçimi
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # İşleme döngüsü
        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        stframe = st.empty()  # Dinamik bir video çerçevesi oluştur
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Modeli kullanarak tahmin yap
            frame = process_frame(frame, model)

            # Çerçeveyi video çıktısına yaz
            out.write(frame)

            # İşlenmiş kareyi Streamlit'te göster
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB formatına dönüştür
            stframe.image(frame_rgb)

            # İlerleme durumu güncelle
            processed_frames += 1
            progress_bar.progress(processed_frames / frame_count)

        # Kaynakları serbest bırak
        cap.release()
        out.release()

        # Kullanıcıya video indirme linki sağlama
        with open(output_video_path, "rb") as video_file:
            st.download_button(label="İndir", data=video_file, file_name="islenmis_video.mp4", mime="video/mp4")

# Kamera ile Canlı Video Çekimi ve Tespit
elif option == "Kamera ile Canlı Video Çekimi ve Tespit":
    cap = None  # Kamera değişkeni, burada tanımlıyoruz

    if "recording" not in st.session_state:
        st.session_state["recording"] = False

    start_button = st.button("Kamerayı Başlat ve Çekime Başla")
    stop_button = st.button("Çekimi Durdur")

    # Start butonuna tıklanırsa kamera başlatılır
    if start_button:
        st.session_state["recording"] = True

    # Stop butonuna tıklanırsa kamera durdurulur
    if stop_button:
        st.session_state["recording"] = False

    if st.session_state["recording"]:
        cap = cv2.VideoCapture(0)  # Kamerayı başlat
        stframe = st.empty()  # Görüntü çerçevesi

        # Video kaydı için ayar
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        width, height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Model ile tahmin yap
            frame = process_frame(frame, model)

            # İşlenmiş görüntüyü göster ve kaydet
            out.write(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Stop butonuna tıklanırsa döngüden çıkılır
            if not st.session_state["recording"]:
                cap.release()
                out.release()
                break

        # Video kaydını bitirip serbest bırakma
        cap.release()
        out.release()

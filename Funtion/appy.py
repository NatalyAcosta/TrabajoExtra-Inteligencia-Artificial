import streamlit as st
import io
import os
from ultralytics import YOLO
import cv2

import shutil
# Muestra el widget para cargar el archivo
archivo = st.file_uploader("Selecciona un archivo de video", type=["mp4","jpg", "png", "jpeg", "gif"])


def ejecutar_yolo(file):
    model = YOLO('Models/best.pt') 
    result = model(file,
    save = True)
    st.image(result[0].plot())
    cv2.imwrite('result.jpg', result[0].plot())
    st.success(" se ha ejecutado con éxito.")
def ejecutar_yolo_V(file):
    model = YOLO('Models/best.pt') 
    result = model(file,
    save = True)
    
    
    st.success("se ha ejecutado con éxito.")


# Muestra el widget para cargar el archivo
# Si el archivo es cargado con éxito, guarda el archivo en una carpeta específica y muestra un mensaje de éxito
if archivo is not None:
    ruta_carpeta = 'runs'
    if os.path.exists(ruta_carpeta):
        shutil.rmtree(ruta_carpeta)
    ruta_carpeta = os.path.abspath("cache")
    with open(os.path.join(ruta_carpeta, archivo.name), "wb") as f:
        f.write(archivo.getbuffer())

    nombre_archivo = archivo.name
    ruta_absoluta = os.path.join(ruta_carpeta, nombre_archivo)
    #st.success(ruta_absoluta.replace("\\", "/"))

    if st.button("Imagen"):
        ejecutar_yolo(ruta_absoluta)

        # Elimina todos los archivos de la carpeta 'cache' después de haber procesado el archivo
        archivos = os.listdir(ruta_carpeta)
        for archivo in archivos:
            os.remove(os.path.join(ruta_carpeta, archivo))
    if st.button("video"):
        ejecutar_yolo_V(ruta_absoluta)

        # Elimina todos los archivos de la carpeta 'cache' después de haber procesado el archivo
        archivos = os.listdir(ruta_carpeta)
        for archivo in archivos:
            ruta_video= 'runs/detect/predict/'+ archivo
            os.remove(os.path.join(ruta_carpeta, archivo))
        
        video = open(ruta_video, "rb")
        video_bytes = video.read()
        st.video(video_bytes,format='video/mp4',start_time=1) 

        #st.success("Todos los archivos han sido eliminados de la carpeta 'cache'")





import requests
import numpy as np
import cv2
import os
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods

@require_http_methods(["GET"])
def capturar_rostro_get(request):
    def capturar_rostro(numero_documento, local_image_paths, face_cascade):
        # Carpeta para almacenar los rostros
        carpeta = os.path.join('dataset', numero_documento)

        # Crear la carpeta si no existea
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
            print("Cree carpeta")

        # Procesar cada ruta de imagen local
        for i, image_path in enumerate(local_image_paths):
            image_path = os.path.join(image_path)
            img = cv2.imread(image_path)
            print("imagen" + image_path)

            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detectar rostros
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Procesar cada rostro detectado
            for (x, y, w, h) in faces:
                # Procesar la región de interés (ROI) del rostro
                roi_gray = gray[y:y+h, x:x+w]

                # Guardar la imagen capturada
                nombre_archivo = f"rostro_{i}.jpg"
                ruta_guardado = os.path.join(carpeta, nombre_archivo)
                print("Ruta de guardado:", ruta_guardado)  # Imprimir la ruta de guardado

                try:
                    cv2.imwrite(ruta_guardado, roi_gray)
                    print(f"Imagen guardada correctamente: {ruta_guardado}")
                except cv2.error as e:
                    print(f"Error al guardar la imagen: {e}")

    # Cargar el clasificador de rostros en cascada de Haar
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Obtener el número de documento y las rutas de las imágenes locales desde la solicitud GET
    numero_documento = request.GET.get("numero_documento", "")  
    local_image_paths = request.GET.getlist("local_image_paths", [])  

    if numero_documento and local_image_paths:
        capturar_rostro(numero_documento, local_image_paths, face_cascade)
        return HttpResponse("Captura de rostros realizada correctamente.")
    else:
        return HttpResponse("Por favor, proporcione tanto el parámetro numero_documento como al menos una ruta de imagen local en la solicitud GET.")
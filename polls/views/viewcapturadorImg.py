import os
import numpy as np
import cv2
import random
import string
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import face_recognition

@csrf_exempt
@require_http_methods(["POST"])
def capturar_rostro(request):

    def guardar_rostro(numero_documento, images):
        # Carpeta para almacenar los rostros
        carpeta = os.path.join('dataset', numero_documento)
        print("Ruta de la carpeta:", carpeta)

        # Crear la carpeta si no existe
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
            print("Carpeta creada.")
        else:
            print("La carpeta ya existe.")

        # Procesar cada imagen
        for i, image in enumerate(images):
            # Leer la imagen desde los datos recibidos
            nparr = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print("Error al leer la imagen.")
                continue

            # Detectar rostros en la imagen
            face_locations = face_recognition.face_locations(img)

            if len(face_locations) == 0:
                print("No se detectaron rostros en la imagen.")
                continue

            # Procesar cada rostro detectado
            for face_location in face_locations:
                top, right, bottom, left = face_location
                rostro = img[top:bottom, left:right]

                # Convertir el rostro a escala de grises
                rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)

                # Generar un código aleatorio de 4 letras
                codigo_aleatorio = ''.join(random.choices(string.ascii_lowercase, k=4))

                # Guardar la imagen del rostro
                nombre_archivo = f"{numero_documento}_{codigo_aleatorio}.png"
                ruta_guardado = os.path.join(carpeta, nombre_archivo)
                try:
                    cv2.imwrite(ruta_guardado, rostro_gray)
                    print(f"Rostro guardado correctamente: {ruta_guardado}")
                except cv2.error as e:
                    print(f"Error al guardar el rostro: {e}")

    # Obtener el número de documento y las imágenes de la solicitud POST
    numero_documento = request.POST.get("numero_documento", "")
    images = request.FILES.getlist("images", [])

    print("Número de documento:", numero_documento)
    print("Imágenes recibidas:", len(images))

    if numero_documento and images:
        guardar_rostro(numero_documento, images)
        return HttpResponse("Captura de rostros realizada correctamente.")
    else:
        return HttpResponse("Por favor, proporcione tanto el parámetro numero_documento como al menos una imagen en la solicitud POST.")

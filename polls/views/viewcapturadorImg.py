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

    def alinear_rostro(img, face_landmarks):
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']

        # Calcular el centro de los ojos
        left_eye_center = np.mean(left_eye, axis=0).astype("int")
        right_eye_center = np.mean(right_eye, axis=0).astype("int")

        # Calcular el ángulo entre los ojos
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx)) - 180

        # Rotar la imagen para alinear los ojos
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
        aligned_face_image = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        return aligned_face_image

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
            face_landmarks_list = face_recognition.face_landmarks(img)

            if len(face_locations) == 0:
                print("No se detectaron rostros en la imagen.")
                continue

            # Procesar cada rostro detectado
            for face_location, face_landmarks in zip(face_locations, face_landmarks_list):
                top, right, bottom, left = face_location
                rostro = img[top:bottom, left:right]

                # Alinear el rostro
                rostro_alineado = alinear_rostro(rostro, face_landmarks)

                # Convertir el rostro a escala de grises
                rostro_gray = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)

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

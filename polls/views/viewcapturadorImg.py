import os
import numpy as np
import cv2
import random
import string
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
import face_recognition
from polls.views.consulta import reiniciar_gunicorn

# Esta vista estará protegida por JWT, se usa el decorador `@permission_classes`
@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])  # Aplica la protección de JWT
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

            # Asegurarse de que la imagen esté en formato RGB
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detectar rostros en la imagen
            face_locations = face_recognition.face_locations(rgb_image)
            face_landmarks_list = face_recognition.face_landmarks(rgb_image)

            if len(face_locations) == 0:
                print("No se detectaron rostros en la imagen.")
                continue

            # Procesar cada rostro detectado
            for face_location, face_landmarks in zip(face_locations, face_landmarks_list):
                top, right, bottom, left = face_location
                rostro = rgb_image[top:bottom, left:right]

                # Redimensionar la imagen a un tamaño específico si es necesario
                rostro_rgb_resized = cv2.resize(rostro, (224, 224))  # Si se desea redimensionar

                # Si necesitas normalizar la imagen en el rango de 0 a 255
                rostro_rgb_resized_normalized = cv2.normalize(rostro_rgb_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Generar un código aleatorio de 4 letras
                codigo_aleatorio = ''.join(random.choices(string.ascii_lowercase, k=4))

                # Guardar la imagen del rostro
                nombre_archivo = f"{numero_documento}_{codigo_aleatorio}.png"
                ruta_guardado = os.path.join(carpeta, nombre_archivo)
                try:
                    cv2.imwrite(ruta_guardado, rostro_rgb_resized_normalized)
                    print(f"Rostro guardado correctamente: {ruta_guardado}")
                except cv2.error as e:
                    print(f"Error al guardar el rostro: {e}")

    # Obtener el número de documento y las imágenes de la solicitud POST
    numero_documento = request.data.get("numero_documento", "")
    images = request.FILES.getlist("images", [])

    print("Número de documento:", numero_documento)
    print("Imágenes recibidas:", len(images))

    if numero_documento and images:
        guardar_rostro(numero_documento, images)
        reiniciar_gunicorn()
        return HttpResponse("Captura de rostros realizada correctamente.")
    else:
        return HttpResponse("Por favor, proporcione tanto el parámetro numero_documento como al menos una imagen en la solicitud POST.")

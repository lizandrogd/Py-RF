import os
import numpy as np
import cv2
import random
import string
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
@require_http_methods(["POST"])
def capturar_rostro(request):
    def capturar_rostro(numero_documento, images, face_cascade):
        # Carpeta para almacenar los rostros
        carpeta = os.path.join('dataset', numero_documento)
        # Imprimir la ruta de la carpeta
        print("Ruta de la carpeta:", carpeta)
        # Crear la carpeta si no existe
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
            print("Cree carpeta")
        else:
            print("La carpeta ya existe")

        # Procesar cada imagen
        for i, image in enumerate(images):
            # Leer la imagen desde los datos recibidos
            nparr = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Generar un código aleatorio de 4 letras
            codigo_aleatorio = ''.join(random.choices(string.ascii_lowercase, k=4))

            # Guardar la imagen procesada por OpenCV
            nombre_archivo_procesada = f"{numero_documento}_{codigo_aleatorio}.png"
            ruta_guardado_procesada = os.path.join(carpeta, nombre_archivo_procesada)
            try:
                cv2.imwrite(ruta_guardado_procesada, gray)
                print(f"Imagen procesada guardada correctamente: {ruta_guardado_procesada}")
            except cv2.error as e:
                print(f"Error al guardar la imagen procesada: {e}")

    # Cargar el clasificador de rostros en cascada de Haar
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Obtener el número de documento y las imágenes de la solicitud POST
    numero_documento = request.POST.get("numero_documento", "")
    images = request.FILES.getlist("images", [])

    if numero_documento and images:
        capturar_rostro(numero_documento, images, face_cascade)
        return HttpResponse("Captura de rostros realizada correctamente.")
    else:
        return HttpResponse("Por favor, proporcione tanto el parámetro numero_documento como al menos una imagen en la solicitud POST.")

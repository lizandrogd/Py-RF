import os
import numpy as np
import cv2
import random
import string
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
import face_recognition
from polls.views.consulta import reiniciar_gunicorn

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def capturar_rostro(request):
    def es_imagen_duplicada(nuevo_encoding, carpeta):
        """ Verifica si la imagen ya existe en la carpeta comparando embeddings. """
        for archivo in os.listdir(carpeta):
            if archivo.endswith(".png"):
                imagen_existente = face_recognition.load_image_file(os.path.join(carpeta, archivo))
                encoding_existente = face_recognition.face_encodings(imagen_existente)

                # Solo comparar si hay un encoding válido
                if encoding_existente:
                    distancia = face_recognition.face_distance(encoding_existente, [nuevo_encoding])
                    if distancia[0] < 0.6:  # Ajustamos el umbral a 0.6 para mayor flexibilidad
                        return True
        return False

    def guardar_rostro(numero_documento, images):
        """ Procesa y almacena rostros detectados. """
        usuario = request.user
        carpeta = os.path.join('dataset', str(usuario), numero_documento)
        os.makedirs(carpeta, exist_ok=True)

        rostros_guardados = 0

        for image in images:
            try:
                nparr = np.frombuffer(image.read(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Error al decodificar la imagen.")

                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)

                if not face_locations:
                    continue  # No se detectó ningún rostro

                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    rostro = rgb_image[top:bottom, left:right]
                    rostro_resized = cv2.resize(rostro, (224, 224))
                    rostro_normalized = cv2.normalize(rostro_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                    # Obtener encoding
                    encoding_nuevo = face_recognition.face_encodings(rostro_resized)
                    
                    # Solo guardar si el encoding es válido y no es una imagen duplicada
                    if encoding_nuevo and len(encoding_nuevo) > 0 and not es_imagen_duplicada(encoding_nuevo[0], carpeta):
                        codigo_aleatorio = ''.join(random.choices(string.ascii_lowercase, k=4))
                        nombre_archivo = f"{numero_documento}_{codigo_aleatorio}.png"
                        ruta_guardado = os.path.join(carpeta, nombre_archivo)

                        cv2.imwrite(ruta_guardado, rostro_normalized)
                        rostros_guardados += 1
            except Exception as e:
                print(f"Error procesando imagen: {e}")  # Para depuración

        return rostros_guardados

    numero_documento = request.data.get("numero_documento", "")
    images = request.FILES.getlist("images", [])

    if not numero_documento or not images:
        return JsonResponse({"error": True, "mensaje": "Debe proporcionar el número de documento y al menos una imagen."})

    rostros_guardados = guardar_rostro(numero_documento, images)

    if rostros_guardados > 0:
        # Solo reiniciar Gunicorn si se guardaron muchas imágenes o si la carpeta estaba vacía
        carpeta_usuario = os.path.join('dataset', str(request.user), numero_documento)
        if rostros_guardados > 5 or not os.listdir(carpeta_usuario):
            reiniciar_gunicorn()
        return JsonResponse({"error": False, "mensaje": f"Se guardaron {rostros_guardados} rostros correctamente."})
    else:
        return JsonResponse({"error": True, "mensaje": "No se detectaron rostros en las imágenes enviadas."})

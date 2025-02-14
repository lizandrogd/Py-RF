import os
import datetime
import numpy as np
import cv2
import random
import string
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
import joblib
import face_recognition

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def reconocimiento_facial(request):
    usuario = request.user
    imagen = request.FILES.get('image')
    documento = request.POST.get('documento')

    if not imagen:
        return JsonResponse({"error": True, "message": "No se proporcionó una imagen."})

    modelo_path = os.path.join('modelos_svm', str(usuario), f'{documento}.pkl')

    if not os.path.exists(modelo_path):
        return JsonResponse({"error": True, "message": "No hay un modelo entrenado para este usuario."})

    try:
        # Procesar la imagen
        nparr = np.frombuffer(imagen.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JsonResponse({"error": True, "message": "Error al procesar la imagen."})

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if not face_encodings:
            return JsonResponse({"error": True, "message": "No se detectó ningún rostro en la imagen."})

        # Cargar modelo y hacer predicción
        modelo_svm = joblib.load(modelo_path)
        prediccion = modelo_svm.predict([face_encodings[0]])

        resultado = "Reconocido" if prediccion[0] == documento else "No reconocido"

        if resultado == "Reconocido":
            # Guardar la imagen en el dataset del usuario
            dataset_usuario = os.path.join('dataset', str(usuario), documento)
            os.makedirs(dataset_usuario, exist_ok=True)

            for face_location in face_locations:
                top, right, bottom, left = face_location
                rostro = rgb_image[top:bottom, left:right]
                rostro_resized = cv2.resize(rostro, (224, 224))
                rostro_normalized = cv2.normalize(rostro_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Nombre de archivo con código aleatorio
                codigo_aleatorio = ''.join(random.choices(string.ascii_lowercase, k=4))
                nombre_archivo = f"{documento}_{codigo_aleatorio}.png"
                ruta_guardado = os.path.join(dataset_usuario, nombre_archivo)

                cv2.imwrite(ruta_guardado, rostro_normalized)

        return JsonResponse({"error": False, "message": resultado})

    except Exception as e:
        return JsonResponse({"error": True, "message": f"Error en la verificación: {e}"})

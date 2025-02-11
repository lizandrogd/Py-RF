from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import joblib
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from polls.models import Log
from polls.views.consulta import procesar_resultados
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes

# Cargar el modelo SVM previamente entrenado
svm_clf = joblib.load('modelo_svm_con_aumento_con_desconocido.pkl')

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])  # Aplica la protección de JWT
def reconocimiento_facial(request):
    tolerance_threshold_svm = 0.60  # Umbral de tolerancia para SVM

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Obtener la imagen de la solicitud
            image_file = request.FILES['image']
            usuario = request.user

            # Cargar la imagen con face_recognition
            image = face_recognition.load_image_file(image_file)

            # Detectar rostros en la imagen
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                return JsonResponse({"error": True, "message": "No se detectaron rostros en la imagen."})

            results = []

            for face_location in face_locations:
                top, right, bottom, left = face_location
                rostro = image[top:bottom, left:right]

                # Redimensionar y normalizar la imagen
                rostro_rgb_resized = cv2.resize(rostro, (224, 224))
                rostro_rgb_resized_normalized = cv2.normalize(
                    rostro_rgb_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                rostro_rgb_resized_normalized_rgb = cv2.cvtColor(
                    rostro_rgb_resized_normalized, cv2.COLOR_BGR2RGB
                )

                # Extraer codificaciones faciales
                face_encodings = face_recognition.face_encodings(rostro_rgb_resized_normalized_rgb)

                if not face_encodings:
                    continue

                face_encoding = face_encodings[0]

                # Predecir con SVM
                svm_scores = svm_clf.decision_function([face_encoding])

                # Normalizar las puntuaciones si es necesario
                if len(svm_scores.shape) == 1:  # Modelo binario
                    svm_probabilities = np.array([svm_scores]).flatten()
                else:  # Modelo multiclase
                    svm_probabilities = svm_scores.flatten()

                # Construir ruta del dataset del usuario
                dataset_path = os.path.join('dataset', str(usuario))

                if not os.path.exists(dataset_path):
                    return JsonResponse({"error": True, "message": "No se encontró el dataset del usuario."})

                # Obtener una lista de las carpetas en el directorio del dataset
                carpetas = sorted([carpeta for carpeta in os.listdir(dataset_path)
                                   if os.path.isdir(os.path.join(dataset_path, carpeta))])

                # Obtener coincidencias que superen el umbral
                svm_matches = [(carpetas[i], svm_conf) for i, svm_conf in enumerate(svm_probabilities)
                               if svm_conf >= tolerance_threshold_svm and i < len(carpetas)]

                if svm_matches:
                    # Ordenar las coincidencias por confianza
                    svm_matches.sort(key=lambda x: x[1], reverse=True)
                    results.extend([match[0] for match in svm_matches])
                else:
                    results.append("Desconocido")

            # Eliminar duplicados y ordenar resultados
            results = sorted(set(results))

            # Guardar logs en la base de datos
            for result in results:
                log_message = f"Rostro reconocido: {result}" if result != "Desconocido" else "Rostro desconocido"
                log_level = "Éxito" if result != "Desconocido" else "Error"
                Log.objects.create(level=log_level, message=log_message, created_at=datetime.now())

            return JsonResponse({"error": False, "results": results})

        except Exception as e:
            return JsonResponse({"error": True, "message": f"Error al procesar la imagen: {e}"})

    return JsonResponse({"error": True, "message": "Debe proporcionar una imagen en una solicitud POST."})

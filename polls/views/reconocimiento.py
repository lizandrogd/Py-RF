from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from polls.models import Log
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes

# Cargar el modelo SVM previamente entrenado
svm_clf = joblib.load('modelo_svm_con_aumento_con_desconocido.pkl')

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])  # Aplica la protección de JWT
def reconocimiento_facial(request):
    tolerance_threshold_svm = 0.50  # Umbral de tolerancia para SVM

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Obtener la imagen y el usuario
            image_file = request.FILES['image']
            usuario = request.user
            documento = request.POST.get('documento')

            # Cargar la imagen
            image = face_recognition.load_image_file(image_file)

            # Detectar rostros
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return JsonResponse({"error": True, "message": "No se detectaron rostros en la imagen."})
        
            dataset_path = os.path.join('dataset', str(usuario))
            if not os.path.exists(dataset_path):
                return JsonResponse({"error": True, "message": "No se encontró el dataset del usuario."})

            carpetas = sorted([carpeta for carpeta in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, carpeta))])

            for face_location in face_locations:
                top, right, bottom, left = face_location
                rostro = image[top:bottom, left:right]

                # Redimensionar y convertir a RGB
                rostro_resized = cv2.resize(rostro, (224, 224))
                rostro_rgb = cv2.cvtColor(rostro_resized, cv2.COLOR_BGR2RGB)

                # Extraer características faciales
                face_encodings = face_recognition.face_encodings(rostro_rgb)
                if not face_encodings:
                    continue

                face_encoding = face_encodings[0]

                # Predecir con SVM
                svm_scores = svm_clf.decision_function([face_encoding])
                svm_probabilities = np.array([svm_scores]).flatten() if len(svm_scores.shape) == 1 else svm_scores.flatten()

                # Verificar coincidencias
                for i, svm_conf in enumerate(svm_probabilities):
                    if svm_conf >= tolerance_threshold_svm and i < len(carpetas):
                        resultado = carpetas[i]

                        # Guardar en el dataset si es reconocido
                        if resultado == documento:
                            log_message = f"Rostro reconocido: {resultado}"
                            log_level = "Éxito"
                            Log.objects.create(level=log_level, message=log_message, created_at=datetime.now())

                            # Guardar la imagen en el dataset
                            user_folder = os.path.join(dataset_path, resultado)
                            os.makedirs(user_folder, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            cv2.imwrite(os.path.join(user_folder, f"{timestamp}.jpg"), rostro_resized)

                            return JsonResponse({"error": False, "results": "reconocido"})

                # Si no hay coincidencia, registrar como desconocido
                Log.objects.create(level="Error", message="Rostro desconocido", created_at=datetime.now())
                return JsonResponse({"error": False, "results": "no reconocido"})

        except Exception as e:
            return JsonResponse({"error": True, "message": f"Error al procesar la imagen: {e}"})

    return JsonResponse({"error": True, "message": "Debe proporcionar una imagen en una solicitud POST."})

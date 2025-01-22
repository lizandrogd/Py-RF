from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import joblib
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime  # Importa datetime
from polls.models import Log  # Importar el modelo Log
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
            
            # Cargar la imagen con face_recognition
            image = face_recognition.load_image_file(image_file)
            
            # Detectar rostros en la imagen
            face_locations = face_recognition.face_locations(image)
            
            # Si se detectan rostros, procesar la imagen
            if face_locations:
                # Inicializar lista para almacenar los resultados
                results = []

                # Procesar cada rostro detectado
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    rostro = image[top:bottom, left:right]
                    
                    # Redimensionar y normalizar la imagen al tamaño esperado (224x224)
                    rostro_rgb_resized = cv2.resize(rostro, (224, 224))
                    rostro_rgb_resized_normalized = cv2.normalize(rostro_rgb_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    # Convertir imagen a RGB
                    rostro_rgb_resized_normalized_rgb = cv2.cvtColor(rostro_rgb_resized_normalized, cv2.COLOR_BGR2RGB)
                    
                    # Extraer codificaciones faciales
                    face_encodings = face_recognition.face_encodings(rostro_rgb_resized_normalized_rgb)
                    
                    # Asegurarse de que al menos se encuentre una codificación
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        
                        # Predecir con SVM
                        svm_scores = svm_clf.decision_function([face_encoding])
                        
                        # Manejo seguro de dimensiones para cálculo de probabilidades
                        if len(svm_scores.shape) == 1:  # Modelo binario
                            svm_probabilities = np.array([np.exp(svm_scores) / np.sum(np.exp(svm_scores), keepdims=True)]).flatten()
                        else:  # Modelo multiclase
                            svm_probabilities = np.exp(svm_scores) / np.sum(np.exp(svm_scores), axis=1, keepdims=True)

                        # Ruta principal del dataset
                        dataset_path = 'dataset/'

                        # Obtener una lista de las carpetas en el directorio del dataset
                        carpetas = os.listdir(dataset_path)

                        # Asegurarse de que solo se consideren directorios y se ordenen alfabéticamente
                        carpetas = sorted([carpeta for carpeta in carpetas if os.path.isdir(os.path.join(dataset_path, carpeta))])

                        # Obtener coincidencias que superen el umbral
                        svm_matches = []

                        for i, svm_conf in enumerate(svm_probabilities):
                            # Imprimir los márgenes de reconocimiento
                            if svm_conf >= tolerance_threshold_svm:
                                if i < len(carpetas):
                                    cedula = carpetas[i]
                                    svm_matches.append((cedula, svm_conf))

                        if svm_matches:
                            # Ordenar las coincidencias
                            svm_matches.sort(key=lambda x: x[1], reverse=True)
                            results.extend([match[0] for match in svm_matches])
                        else:
                            results.append("Desconocido")

                results = eliminar_duplicados(results)

                # Guardar logs en la base de datos
                for result in results:
                    if result == "Desconocido":
                        Log.objects.create(level="Error", message="Rostro desconocido", created_at=datetime.now())
                    else:
                        Log.objects.create(level="Éxito", message=f"Rostro reconocido: {result}", created_at=datetime.now())
                
                return procesar_resultados(results)
            
            else:
                return HttpResponseBadRequest("No se detectaron rostros en la imagen")
        
        except Exception as e:
            return HttpResponseBadRequest(f"Error al procesar la imagen: {e}")
    
    else:
        return HttpResponseBadRequest("Debe proporcionar una imagen en una solicitud POST.")
    
def eliminar_duplicados(results):
    # Convertimos la lista en un conjunto para eliminar duplicados
    unique_results = list(set(results))
    
    # Ordenamos la lista final de resultados para mantener un orden consistente
    unique_results.sort()
    
    return unique_results

from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import joblib
import cv2
import face_recognition
import numpy as np
import os

from polls.views.consulta import procesar_resultados
from polls.views.consulta import guardar_rostro_desconocido

# Load trained models (SVM only)
svm_clf = joblib.load('modelo_svm_con_aumento_con_desconocido.pkl')

@csrf_exempt
def reconocimiento_facial(request):
    tolerance_threshold_svm = 0.63  # Umbral de tolerancia para SVM

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Get image from request
            image_file = request.FILES['image']
            
            # Load image with face_recognition
            image = face_recognition.load_image_file(image_file)
            
            # Detect faces in the image
            face_locations = face_recognition.face_locations(image)
            
            # If faces are detected, process the image
            if face_locations:
                # Initialize list to store results
                results = []

                # Process each detected face
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
                        svm_probabilities = np.exp(svm_scores) / np.sum(np.exp(svm_scores), axis=1, keepdims=True)

                        # Ruta principal del dataset
                        dataset_path = 'dataset/'

                        # Obtener una lista de las carpetas en el directorio del dataset
                        carpetas = os.listdir(dataset_path)

                        # Asegurarse de que solo se consideren directorios y se ordenen alfabéticamente
                        carpetas = sorted([carpeta for carpeta in carpetas if os.path.isdir(os.path.join(dataset_path, carpeta))])

                        # Obtener coincidencias que superen el umbral
                        svm_matches = []

                        for i, svm_conf in enumerate(svm_probabilities[0]):
                            print(f"Rostro {i + 1}: SVM - {svm_conf * 100:.2f}%")
                            
                            # Imprimir los márgenes de reconocimiento
                            if svm_conf >= tolerance_threshold_svm:
                                if i < len(carpetas):
                                    cedula = carpetas[i]
                                    svm_matches.append((cedula, svm_conf))
                                    
                                    # Imprimir información de coincidencia
                                    print(f"Coincidencia encontrada: {cedula} con SVM: {svm_conf * 100:.2f}%")

                        if svm_matches:
                            # Ordenar las coincidencias
                            svm_matches.sort(key=lambda x: x[1], reverse=True)
                            results.extend([match[0] for match in svm_matches])
                        else:
                            results.append("Desconocido")
                            guardar_rostro_desconocido(rostro_rgb_resized_normalized_rgb)

                results = eliminar_duplicados(results)
                print(f"Results: {results}")
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


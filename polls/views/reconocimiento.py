from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import joblib
import cv2
import face_recognition
import numpy as np
from sklearn.svm import SVC
import os

from polls.views.consulta import procesar_resultados

# Load trained models (KNN and SVM)
knn_clf = joblib.load('modelo_knn_con_aumento_con_desconocido.pkl')
svm_clf = joblib.load('modelo_svm_con_aumento_con_desconocido.pkl')

@csrf_exempt
def reconocimiento_facial(request):
    tolerance_threshold_knn = 0.66  # Umbral de tolerancia
    tolerance_threshold_svm = 0.64

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
                        
                        # Predecir con KNN
                        knn_prediction = knn_clf.predict_proba([face_encoding])
                        
                        # Predecir con SVM
                        svm_scores = svm_clf.decision_function([face_encoding])
                        svm_probabilities = np.exp(svm_scores) / np.sum(np.exp(svm_scores), axis=1, keepdims=True)

                        # Ruta principal del dataset
                        dataset_path = 'dataset/'

                        # Obtener una lista de las carpetas en el directorio del dataset
                        carpetas = os.listdir(dataset_path)

                        # Asegurarse de que solo se consideren directorios y se ordenen alfabéticamente
                        carpetas = sorted([carpeta for carpeta in carpetas if os.path.isdir(os.path.join(dataset_path, carpeta))])

                        # Obtener todas las coincidencias que superen el umbral
                        knn_matches = []
                        svm_matches = []

                        for i, (knn_conf, svm_conf) in enumerate(zip(knn_prediction[0], svm_probabilities[0])):
                            if knn_conf >= tolerance_threshold_knn and svm_conf >= tolerance_threshold_svm:
                                if i < len(carpetas):
                                    cedula = carpetas[i]
                                    knn_matches.append((cedula, knn_conf))
                                    svm_matches.append((cedula, svm_conf))

                        if knn_matches or svm_matches:
                            # Combinar y ordenar las coincidencias
                            all_matches = list(set(knn_matches + svm_matches))
                            all_matches.sort(key=lambda x: x[1], reverse=True)
                            results.append([match[0] for match in all_matches])
                        else:
                            results.append(["Desconocido"])
                
                # Eliminar duplicados antes de procesar los resultados
                results_unicos = eliminar_duplicados(results)

                print(f"Results: {results_unicos}")
                return procesar_resultados(results_unicos)
            
            else:
                return HttpResponseBadRequest("No se detectaron rostros en la imagen")
        
        except Exception as e:
            return HttpResponseBadRequest(f"Error al procesar la imagen: {e}")
    
    else:
        return HttpResponseBadRequest("Debe proporcionar una imagen en una solicitud POST.")
    
def eliminar_duplicados(results):
    # Crear un conjunto para almacenar las cédulas únicas
    cedulas_unicas = set()
    # Iterar sobre los resultados y agregar solo cédulas únicas al conjunto
    for cedula in results:
        # Convertir cada cédula a una tupla si es una lista, de lo contrario usar tal cual
        cedula_hashable = tuple(cedula) if isinstance(cedula, list) else cedula
        cedulas_unicas.add(cedula_hashable)
    # Convertir el conjunto de nuevo a una lista (de listas si corresponde)
    return [list(c) if isinstance(c, tuple) else c for c in cedulas_unicas]
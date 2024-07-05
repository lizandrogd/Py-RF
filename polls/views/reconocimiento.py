from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import joblib
import cv2
import face_recognition
import numpy as np
from sklearn.svm import SVC

from polls.views.consulta import procesar_resultados

# Load trained models (KNN and SVM)
knn_clf = joblib.load('modelo_knn_con_aumento_con_desconocido.pkl')
svm_clf = joblib.load('modelo_svm_con_aumento_con_desconocido.pkl')

@csrf_exempt
def reconocimiento_facial(request):
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
                    
                    # Convert image to RGB
                    rostro_rgb_resized_normalized_rgb = cv2.cvtColor(rostro_rgb_resized_normalized, cv2.COLOR_BGR2RGB)
                    
                    # Extract facial encodings
                    face_encodings = face_recognition.face_encodings(rostro_rgb_resized_normalized_rgb)
                    
                    # Ensure that at least one encoding is found
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        
                        # Use KNN to predict the label
                        knn_prediction = knn_clf.predict_proba([face_encoding])
                        knn_name = knn_clf.classes_[np.argmax(knn_prediction)]
                        print(f"KNN Prediction: {knn_name}")
                        
                        # Use SVM to predict the label
                        svm_scores = svm_clf.decision_function([face_encoding])
                        svm_probabilities = np.exp(svm_scores) / np.sum(np.exp(svm_scores), axis=1, keepdims=True)
                        svm_name = svm_clf.classes_[np.argmax(svm_probabilities)]
                        print(f"SVM Prediction: {svm_name}")
                        
                        # Check if KNN and SVM predictions are equal
                        if knn_name == svm_name:
                            results.append(str(knn_name))
                        else:
                            results.append("Desconocido")
                
                # Process results as needed (here using a function procesar_resultados)
                print(f"Results: {results}")
                return procesar_resultados(results)
            
            else:
                return HttpResponseBadRequest("No faces detected in the image")
        
        except Exception as e:
            return HttpResponseBadRequest(f"Error processing image: {e}")
    
    else:
        return HttpResponseBadRequest("You must provide an image in a POST request.")

from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import joblib
import face_recognition
import numpy as np
from sklearn.svm import SVC  # Import SVC from scikit-learn

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
            
            # Resize image to 224x224 and convert to RGB
            resized_image = face_recognition.resize_image(image, (224, 224))
            rgb_image = resized_image.astype(np.uint8)
            
            # Detect faces in the image
            face_locations = face_recognition.face_locations(rgb_image)
            
            # If faces are detected, process the image
            if face_locations:
                # Extract facial encodings from detected faces
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                # Initialize list to store results
                results = []
                
                # Iterate over each facial encoding found
                for face_encoding in face_encodings:
                    # Use KNN to predict the label
                    knn_prediction = knn_clf.predict_proba([face_encoding])
                    knn_name = knn_clf.classes_[np.argmax(knn_prediction)]
                    
                    # Use SVM to predict the label
                    svm_scores = svm_clf.decision_function([face_encoding])
                    svm_probabilities = np.exp(svm_scores) / np.sum(np.exp(svm_scores), axis=1, keepdims=True)
                    svm_name = svm_clf.classes_[np.argmax(svm_probabilities)]
                    
                    # Check if KNN and SVM predictions are equal
                    if knn_name != svm_name:
                        # Add predicted label only if both predictions are equal
                        results.append(str(knn_name))
                    else:
                        # If predictions are not equal, add "Unknown"
                        results.append("Desconocido")
                
                # Process results as needed (here using a function procesar_resultados)
                return procesar_resultados(results)
            
            else:
                return HttpResponseBadRequest("No faces detected in the image")
        
        except Exception as e:
            return HttpResponseBadRequest(f"Error processing image: {e}")
    
    else:
        return HttpResponseBadRequest("You must provide an image in a POST request.")

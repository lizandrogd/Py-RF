import cv2
import numpy as np
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from polls.views.consulta import procesar_resultados

@csrf_exempt
def reconocimiento_facial(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Obtener la imagen de la solicitud POST
        image = request.FILES['image']
        
        # Cargar los modelos entrenados (KNN y SVM)
        knn_clf = joblib.load('modelo_knn_con_aumento_con_desconocido.pkl')
        svm_clf = joblib.load('modelo_svm_con_aumento_con_desconocido.pkl')

        # Convertir la imagen a una matriz numpy
        nparr = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mejorar el contraste de la imagen
        gray = cv2.equalizeHist(gray)

        # Inicializar el clasificador de rostros en cascada de Haar
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        resultados = []

        # Detectar rostros en la imagen con parámetros ajustados
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray_resized = cv2.resize(roi_gray, (100, 100))

            # Aplicar preprocesamiento
            roi_gray_resized = roi_gray_resized.astype('float32') / 255.0
            roi_gray_resized_flattened = roi_gray_resized.flatten().reshape(1, -1)
            
            # Hacer predicción con KNN
            knn_prob = knn_clf.predict_proba(roi_gray_resized_flattened)
            knn_label = knn_clf.classes_[np.argmax(knn_prob)]
            knn_confidence = np.max(knn_prob)
            
            # Hacer predicción con SVM
            svm_prob = svm_clf.decision_function(roi_gray_resized_flattened)
            svm_label = svm_clf.classes_[np.argmax(svm_prob)]
            svm_confidence = np.max(svm_prob)

            # Verificar si las etiquetas predichas coinciden en ambos modelos y si superan un umbral de confianza
            confidence_threshold = 0.9  # Aumentar umbral de confianza para mayor rigurosidad
            if knn_label == svm_label and knn_confidence >= confidence_threshold and svm_confidence >= confidence_threshold:
                resultados.append(str(knn_label))
            else:
                resultados.append("Desconocido")

        # Retornar los resultados como un JSON
        return procesar_resultados(resultados)
    else:
        return JsonResponse({"error": "Debe proporcionar una imagen en la solicitud POST."})

import cv2
import numpy as np
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import face_recognition
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

        # Detectar rostros en la imagen
        face_locations = face_recognition.face_locations(img)
        face_landmarks_list = face_recognition.face_landmarks(img)

        resultados = []

        for face_location, face_landmarks in zip(face_locations, face_landmarks_list):
            top, right, bottom, left = face_location
            face_image = img[top:bottom, left:right]

            # Alinear el rostro
            rostro_alineado = alinear_rostro(face_image, face_landmarks)

            # Convertir el rostro a escala de grises y redimensionar si es necesario
            rostro_gray = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
            rostro_gray_resized = cv2.resize(rostro_gray, (100, 100))  # Redimensionar según necesidad

            # Aplicar preprocesamiento adicional si es necesario
            # rostro_gray_resized = rostro_gray_resized.astype('float32') / 255.0  # Normalización

            # Hacer predicción con KNN
            knn_prob = knn_clf.predict_proba(rostro_gray_resized.flatten().reshape(1, -1))
            knn_label = knn_clf.classes_[np.argmax(knn_prob)]
            knn_confidence = np.max(knn_prob)

            # Hacer predicción con SVM
            svm_prob = svm_clf.decision_function(rostro_gray_resized.flatten().reshape(1, -1))
            svm_label = svm_clf.classes_[np.argmax(svm_prob)]
            svm_confidence = np.max(svm_prob)

            # Verificar si las etiquetas predichas coinciden en ambos modelos y si superan un umbral de confianza
            confidence_threshold = 0.90  # Umbral de confianza
            if knn_label == svm_label and knn_confidence >= confidence_threshold and svm_confidence >= confidence_threshold:
                resultados.append(str(knn_label))
            else:
                resultados.append("Desconocido")

        # Procesar los resultados obtenidos
        return procesar_resultados(resultados)
    else:
        return JsonResponse({"error": "Debe proporcionar una imagen en la solicitud POST."})

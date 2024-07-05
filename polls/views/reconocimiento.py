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

        # Mejorar el contraste de la imagen
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detectar rostros en la imagen
        face_locations = face_recognition.face_locations(img, model='cnn')  # Usar el modelo 'cnn' puede ser más robusto

        resultados = []

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = img[top:bottom, left:right]

            # Convertir el rostro a escala de grises y redimensionar
            face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image_resized = cv2.resize(face_image_gray, (100, 100))

            # Aplicar preprocesamiento
            face_image_resized = face_image_resized.astype('float32') / 255.0
            face_image_resized_flattened = face_image_resized.flatten().reshape(1, -1)

            # Hacer predicción con KNN
            knn_prob = knn_clf.predict_proba(face_image_resized_flattened)
            knn_label = knn_clf.classes_[np.argmax(knn_prob)]
            knn_confidence = np.max(knn_prob)

            # Hacer predicción con SVM
            svm_prob = svm_clf.decision_function(face_image_resized_flattened)
            svm_label = svm_clf.classes_[np.argmax(svm_prob)]
            svm_confidence = np.max(svm_prob)

            # Verificar si las etiquetas predichas coinciden en ambos modelos y si superan un umbral de confianza
            confidence_threshold = 0.95  # Ajusta el umbral según la precisión que desees
            if knn_label == svm_label and knn_confidence >= confidence_threshold and svm_confidence >= confidence_threshold:
                resultados.append(str(knn_label))
            else:
                resultados.append("Desconocido")

        # Asegúrate de procesar todos los resultados obtenidos
        return procesar_resultados(resultados)
    else:
        return JsonResponse({"error": "Debe proporcionar una imagen en la solicitud POST."})

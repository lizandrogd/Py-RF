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

            # Obtener las ubicaciones de los ojos
            left_eye = face_landmarks['left_eye']
            right_eye = face_landmarks['right_eye']

            # Calcular el centro de los ojos
            left_eye_center = np.mean(left_eye, axis=0).astype("int")
            right_eye_center = np.mean(right_eye, axis=0).astype("int")

            # Calcular el ángulo entre los ojos
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx)) - 180

            # Rotar la imagen para alinear los ojos
            eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                           (left_eye_center[1] + right_eye_center[1]) // 2)
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
            aligned_face_image = cv2.warpAffine(face_image, M, (face_image.shape[1], face_image.shape[0]))

            # Convertir el rostro a escala de grises y redimensionar
            face_image_gray = cv2.cvtColor(aligned_face_image, cv2.COLOR_BGR2GRAY)
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
            confidence_threshold = 0.90  # Aumentar umbral de confianza para mayor rigurosidad
            if knn_label == svm_label and knn_confidence >= confidence_threshold and svm_confidence >= confidence_threshold:
                resultados.append(str(knn_label))
            else:
                resultados.append("Desconocido")

        # Asegúrate de procesar todos los resultados obtenidos
        return procesar_resultados(resultados)
    else:
        return JsonResponse({"error": "Debe proporcionar una imagen en la solicitud POST."})

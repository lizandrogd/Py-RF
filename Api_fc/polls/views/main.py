import cv2
import numpy as np
import joblib
from django.http import JsonResponse

def reconocimiento_facial(request):
    # Cargar el modelo entrenado
    knn_clf = joblib.load('modelo_con_aumento_con_desconocido.pkl')

    # Iniciar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)

    # Inicializar el clasificador de rostros en cascada de Haar
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    resultados = []

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray_resized = cv2.resize(roi_gray, (100, 100))
            label, confianza = knn_clf.predict(roi_gray_resized.flatten().reshape(1, -1))[0], knn_clf.predict_proba(roi_gray_resized.flatten().reshape(1, -1))[0]
            
            # Obtener la etiqueta y la probabilidad máxima
            etiqueta_max_prob = np.argmax(confianza)
            max_prob = confianza[etiqueta_max_prob]

            # Verificar si el rostro fue reconocido o es desconocido
            if max_prob < 0.8:
                # Rostro desconocido
                resultados.append("Desconocido")
            else:
                # Rostro reconocido
                resultados.append(str(label))

        # Romper el bucle para enviar la respuesta después de un solo marco
        break

    cap.release()
    cv2.destroyAllWindows()

    # Retornar la respuesta como un JSON
    return JsonResponse({"resultados": resultados})
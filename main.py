import cv2
import numpy as np
import joblib

def reconocimiento_facial():
    # Cargar el modelo entrenado
    knn_clf = joblib.load('modelo_con_aumento_con_desconocido.pkl')

    # Iniciar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)

    # Inicializar el clasificador de rostros en cascada de Haar
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
                # Dibujar el rectángulo en rojo para un rostro desconocido
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Añadir la etiqueta "Desconocido" debajo del rectángulo
                cv2.putText(frame, "Desconocido", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # Dibujar el rectángulo en azul para un rostro reconocido
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Añadir la etiqueta numérica debajo del rectángulo
                cv2.putText(frame, str(label), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Reconocimiento Facial en Tiempo Real', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Realizar el reconocimiento facial
reconocimiento_facial()
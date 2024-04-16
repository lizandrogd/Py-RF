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
        
        # Cargar el modelo entrenado
        knn_clf = joblib.load('modelo_con_aumento_con_desconocido.pkl')

        # Convertir la imagen a una matriz numpy
        nparr = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Inicializar el clasificador de rostros en cascada de Haar
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        resultados = []

        # Detectar rostros en la imagen
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray_resized = cv2.resize(roi_gray, (100, 100))

            # Aplicar preprocesamiento
            roi_gray_resized = roi_gray_resized.astype('float32') / 255.0
            
            # Hacer predicción
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

        # Retornar los resultados como un JSON
        return procesar_resultados(resultados)
    else:
        return JsonResponse({"error": "Debe proporcionar una imagen en la solicitud POST."})

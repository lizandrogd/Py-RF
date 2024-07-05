from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import joblib
import face_recognition
import numpy as np

from polls.views.consulta import procesar_resultados

# Cargar los modelos entrenados (KNN y SVM)
knn_clf = joblib.load('modelo_knn_con_aumento_con_desconocido.pkl')
svm_clf = joblib.load('modelo_svm_con_aumento_con_desconocido.pkl')

@csrf_exempt
def reconocimiento_facial(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Obtener la imagen del request
            image_file = request.FILES['image']
            
            # Leer la imagen con face_recognition
            image = face_recognition.load_image_file(image_file)
            
            # Detectar rostros en la imagen
            face_locations = face_recognition.face_locations(image)
            
            # Si se detectan rostros, procesar la imagen
            if face_locations:
                # Extraer encodings faciales de los rostros detectados
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                # Inicializar lista para almacenar resultados
                results = []
                
                # Iterar sobre cada encoding facial encontrado
                for face_encoding in face_encodings:
                    # Utilizar KNN para predecir la etiqueta
                    knn_prediction = knn_clf.predict_proba([face_encoding])
                    knn_name = knn_clf.classes_[np.argmax(knn_prediction)]
                    print(knn_prediction)
                    # Utilizar SVM para predecir la etiqueta
                    svm_prediction = svm_clf.predict_proba([face_encoding])
                    svm_name = svm_clf.classes_[np.argmax(svm_prediction)]
                    print(svm_prediction)
                    # Verificar si las predicciones de KNN y SVM son iguales
                    if knn_name != svm_name:
                        # Agregar la etiqueta predicha solo si ambas predicciones son iguales
                        results.append(str(knn_name))
                    else:
                        # Si las predicciones no son iguales, agregar "Desconocido"
                        results.append("Desconocido")
                        print (knn_name)
                # Procesar los resultados como necesites (aquí se usa una función procesar_resultados)
                return procesar_resultados(results)
            
            else:
                return HttpResponseBadRequest({"No se detectaron rostros en la imagen"})
        
        except Exception as e:
            return HttpResponseBadRequest({"Error al procesar la imagen"})
    
    else:
        return HttpResponseBadRequest("Debe proporcionar una imagen en la solicitud POST.")

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import face_recognition
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
                    knn_prediction = knn_clf.predict([face_encoding])
                    
                    # Utilizar SVM para predecir la etiqueta
                    svm_prediction = svm_clf.predict([face_encoding])
                    
                    # Verificar si las predicciones de KNN y SVM son iguales
                    if knn_prediction[0] == svm_prediction[0]:
                        # Agregar la etiqueta predicha solo si ambas predicciones son iguales
                        results.append(str(knn_prediction[0]))
                    else:
                        # Si las predicciones no son iguales, agregar "Desconocido"
                        results.append("Desconocido")
                
                # Función para procesar los resultados (puedes ajustar según tu necesidad)
                        return procesar_resultados(results)

            
            else:
                 return JsonResponse({"error": "No se dtectaron tostros en la imagen"})
        
        except Exception as e:            
            return JsonResponse({"error":  "Error al procesar la imagen"})
    
    else:
         return JsonResponse({"error": "Debe proporcionar una imagen en la solicitud POST."})

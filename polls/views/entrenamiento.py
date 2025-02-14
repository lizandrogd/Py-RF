import os
import numpy as np
import logging
from collections import Counter
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import face_recognition
from polls.views.consulta import reiniciar_gunicorn

logger = logging.getLogger(__name__)

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])  
def entrenamiento(request):
    usuario = request.user
    documento = request.POST.get('documento')

    dataset_path = os.path.join('dataset', str(usuario), str(documento))

    if not os.path.exists(dataset_path):
        return JsonResponse({"error": True, "message": f"No se encontró el dataset en {dataset_path}."})

    images = []
    labels = []

    for filename in sorted(os.listdir(dataset_path)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(dataset_path, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if len(face_encodings) == 1:
                    images.append(face_encodings[0])
                    labels.append(documento)  # Se etiqueta con el documento del usuario
                elif len(face_encodings) > 1:
                    logger.warning(f"Más de un rostro detectado en {image_path}. Ignorando esta imagen.")

            except Exception as e:
                logger.error(f"Error al procesar la imagen {image_path}: {e}")

    if not images or not labels:
        return JsonResponse({"error": True, "message": "No se han cargado imágenes o etiquetas para el entrenamiento."})

    X = np.array(images)
    y = np.array(labels)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if len(set(y_encoded)) < 2:
        return JsonResponse({"error": True, "message": "Debe haber al menos dos imágenes para el entrenamiento."})

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)

    svm_classifier = SVC(kernel='linear', C=1.0, gamma='scale')
    svm_classifier.fit(X_train, y_train)

    modelo_path = os.path.join('modelos_svm', str(usuario), str(documento))
    os.makedirs(modelo_path, exist_ok=True)

    joblib.dump(svm_classifier, os.path.join(modelo_path, 'svm.pkl'))

    reiniciar_gunicorn()

    return JsonResponse({
        "error": False,
        "message": "Entrenamiento completado.",
        "num_images": len(images),
        "num_profiles": 1,
    })

import os
import numpy as np
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

@csrf_exempt
@api_view(['GET'])
@permission_classes([IsAuthenticated])  # Aplica la protección de JWT
def entrenamiento(request):
    usuario = request.user
    dataset_path = os.path.join('dataset', str(usuario))  # Corrige la ruta del dataset

    if not os.path.exists(dataset_path):
        return JsonResponse({"error": True, "message": "No se encontró el dataset del usuario."})

    images = []
    labels = []

    for foldername in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, foldername)
        if os.path.isdir(folder_path):
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder_path, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)
                        if len(face_encodings) == 1:
                            images.append(face_encodings[0])
                            labels.append(foldername)
                        elif len(face_encodings) > 1:
                            print(f"Más de un rostro detectado en {image_path}. Ignorando esta imagen.")
                    except Exception as e:
                        print(f"Error al procesar la imagen {image_path}: {e}")

    if not images or not labels:
        return JsonResponse({"error": True, "message": "No se han cargado imágenes o etiquetas para el entrenamiento."})

    X = np.array(images)
    y = np.array(labels)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_counts = Counter(y_encoded)
    print("Clases únicas en y_encoded:", set(y_encoded))
    print("Distribución de clases antes de train_test_split:", class_counts)

    if any(count < 2 for count in class_counts.values()):
        return JsonResponse({"error": True, "message": "Cada clase debe tener al menos dos imágenes para el entrenamiento."})

    min_test_size = len(set(y_encoded))
    test_size = max(0.2, min_test_size / len(y_encoded))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    print("Clases únicas en y_train después de train_test_split:", set(y_train))
    print("Distribución en y_train:", Counter(y_train))
    print("Distribución en y_test:", Counter(y_test))

    if len(set(y_train)) < 2:
        return JsonResponse({"error": True, "message": "El conjunto de entrenamiento debe tener al menos dos clases."})

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)

    svm_classifier = SVC(kernel='linear', C=1.0, gamma='scale')
    svm_classifier.fit(X_train, y_train)

    knn_model_filename = 'modelo_knn_con_aumento_con_desconocido.pkl'
    svm_model_filename = 'modelo_svm_con_aumento_con_desconocido.pkl'

    joblib.dump(knn_classifier, knn_model_filename)
    joblib.dump(svm_classifier, svm_model_filename)

    knn_accuracy = knn_classifier.score(X_test, y_test)
    svm_accuracy = svm_classifier.score(X_test, y_test)

    class_distribution = {label: count for label, count in Counter(labels).items()}

    reiniciar_gunicorn()

    return JsonResponse({
        "error": False,
        "message": "Entrenamiento completado.",
        "knn_accuracy": round(knn_accuracy, 2),
        "svm_accuracy": round(svm_accuracy, 2),
        "num_images": len(images),
        "num_profiles": len(set(labels)),
        "class_distribution": class_distribution
    })

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import face_recognition
from django.http import HttpResponse

def entrenamiento(request):
    dataset_path = 'dataset/'

    images = []
    labels = []

    for foldername in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, foldername)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0:
                    images.append(face_encodings[0])
                    labels.append(foldername)
                    print("Etiqueta", labels)

    X = np.array(images)
    y = np.array(labels)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)

    svm_classifier = SVC(kernel='linear', C=1.0, gamma='scale', probability=True)
    svm_classifier.fit(X_train, y_train)

    knn_model_filename = 'modelo_knn_con_aumento_con_desconocido.pkl'
    svm_model_filename = 'modelo_svm_con_aumento_con_desconocido.pkl'
    label_encoder_filename = 'label_encoder.pkl'

    joblib.dump(knn_classifier, knn_model_filename)
    joblib.dump(svm_classifier, svm_model_filename)
    joblib.dump(label_encoder, label_encoder_filename)

    knn_accuracy = knn_classifier.score(X_test, y_test)
    svm_accuracy = svm_classifier.score(X_test, y_test)

    num_imagenes = len(images)
    num_perfiles = len(set(labels))

    response = (
        f"Entrenamiento completado.<br>"
        f"KNN Accuracy: {knn_accuracy:.2f}<br>"
        f"SVM Accuracy: {svm_accuracy:.2f}<br>"
        f"Número de imágenes entrenadas: {num_imagenes}<br>"
        f"Número de perfiles entrenados: {num_perfiles}"
    )

    return HttpResponse(response)

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
    # Ruta del directorio principal del dataset
    dataset_path = 'dataset/'

    # Listas para almacenar imágenes y etiquetas
    images = []
    labels = []

    # Recorremos cada carpeta dentro del directorio principal en orden alfabético
    for foldername in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, foldername)
        if os.path.isdir(folder_path):
            # Recorremos cada archivo de imagen dentro de la carpeta en orden alfabético
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filtramos solo archivos de imagen
                    image_path = os.path.join(folder_path, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)
                        
                        # Si hay al menos un rostro en la imagen, agregamos la imagen y la etiqueta
                        if len(face_encodings) > 0:
                            images.append(face_encodings[0])  # Tomamos el primer encoding como características
                            labels.append(foldername)  # Usamos el nombre de la carpeta como etiqueta
                    except Exception as e:
                        print(f"Error al procesar la imagen {image_path}: {e}")

    # Convertimos a numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Codificamos las etiquetas como números
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Dividimos los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Entrenamiento de KNN
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)

    # Entrenamiento de SVM
    svm_classifier = SVC(kernel='linear', C=1.0, gamma='scale')
    svm_classifier.fit(X_train, y_train)

    # Guardar modelos entrenados
    knn_model_filename = 'modelo_knn_con_aumento_con_desconocido.pkl'
    svm_model_filename = 'modelo_svm_con_aumento_con_desconocido.pkl'

    # Guardar modelos en archivos .pkl
    joblib.dump(knn_classifier, knn_model_filename)
    joblib.dump(svm_classifier, svm_model_filename)

    # Calcular accuracies
    knn_accuracy = knn_classifier.score(X_test, y_test)
    svm_accuracy = svm_classifier.score(X_test, y_test)

    # Obtener números de imágenes y perfiles
    num_imagenes = len(images)
    num_perfiles = len(set(labels))

    # Construir la respuesta
    response = (
        f"Entrenamiento completado.<br>"
        f"KNN Accuracy: {knn_accuracy:.2f}<br>"
        f"SVM Accuracy: {svm_accuracy:.2f}<br>"
        f"Número de imágenes entrenadas: {num_imagenes}<br>"
        f"Número de perfiles entrenados: {num_perfiles}"
    )

    return HttpResponse(response)

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
import joblib 
from django.http import HttpResponse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

def cargar_imagenes_y_etiquetas(ruta_dataset):
    imagenes = []
    etiquetas = []

    for nombre_persona in os.listdir(ruta_dataset):
        carpeta_persona = os.path.join(ruta_dataset, nombre_persona)

        for imagen_nombre in os.listdir(carpeta_persona):
            imagen_ruta = os.path.join(carpeta_persona, imagen_nombre)
            imagen = cv2.imread(imagen_ruta, cv2.IMREAD_GRAYSCALE)
            imagen = cv2.resize(imagen, (100, 100))
            imagenes.append(imagen)
            etiquetas.append(nombre_persona)

            # Aumento de datos
            imagen_flip = cv2.flip(imagen, 1)
            imagenes.append(imagen_flip)
            etiquetas.append(nombre_persona)

            rows, cols = imagen.shape
            for angulo in range(-20, 21, 5):
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angulo, 1)
                imagen_rotada = cv2.warpAffine(imagen, M, (cols, rows))
                imagenes.append(imagen_rotada)
                etiquetas.append(nombre_persona)

            imagen_desenfocada = cv2.GaussianBlur(imagen, (5, 5), 0)
            imagenes.append(imagen_desenfocada)
            etiquetas.append(nombre_persona)

            mean = 0
            var = 0.1 * 255
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, imagen.shape)
            imagen_con_ruido = imagen + gaussian
            imagenes.append(imagen_con_ruido)
            etiquetas.append(nombre_persona)

    # Añadir imagen en blanco para "Desconocido"
    imagenes.append(np.zeros((100, 100), dtype=np.uint8))
    etiquetas.append("Desconocido")

    # Guardar etiquetas
    np.save('etiquetas.npy', etiquetas)

    return imagenes, etiquetas

def entrenamiento(request):
    ruta_dataset = 'dataset/'
    imagenes, etiquetas = cargar_imagenes_y_etiquetas(ruta_dataset)

    imagenes = np.array(imagenes)
    etiquetas = np.array(etiquetas)

    imagenes = imagenes.astype('float32') / 255.0

    X_train, X_test, y_train, y_test = train_test_split(imagenes.reshape(len(imagenes), -1), etiquetas, test_size=0.2, random_state=42)

    # Entrenamiento y evaluación del clasificador KNN
    knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    knn_clf.fit(X_train, y_train)
    knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=5)
    knn_accuracy = accuracy_score(y_test, knn_clf.predict(X_test))

    # Guardar modelo KNN
    joblib.dump(knn_clf, 'modelo_knn_con_aumento_con_desconocido.pkl')

    # Entrenamiento y evaluación del clasificador SVM
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_clf.fit(X_train, y_train)
    svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=5)
    svm_accuracy = accuracy_score(y_test, svm_clf.predict(X_test))

    # Guardar modelo SVM
    joblib.dump(svm_clf, 'modelo_svm_con_aumento_con_desconocido.pkl')

    response = f"Entrenamiento completado.<br>KNN Accuracy: {knn_accuracy:.2f}<br>SVM Accuracy: {svm_accuracy:.2f}"

    return HttpResponse(response)

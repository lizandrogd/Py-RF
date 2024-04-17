import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
import joblib 
from django.http import HttpResponse

def cargar_imagenes_y_etiquetas(ruta_dataset):
    imagenes = []
    etiquetas = []

    for nombre_persona in os.listdir(ruta_dataset):
        carpeta_persona = os.path.join(ruta_dataset, nombre_persona)

        for imagen_nombre in os.listdir(carpeta_persona):
            imagen_ruta = os.path.join(carpeta_persona, imagen_nombre)
            imagen = cv2.imread(imagen_ruta, cv2.IMREAD_GRAYSCALE)
            # Redimensionar la imagen a un tamaño fijo (por ejemplo, 100x100 píxeles)
            imagen = cv2.resize(imagen, (100, 100))
            imagenes.append(imagen)
            etiquetas.append(nombre_persona)

            # Aplicar aumento de datos a la imagen
            imagen_flip = cv2.flip(imagen, 1)  # Volteo horizontal
            imagenes.append(imagen_flip)
            etiquetas.append(nombre_persona)

            # Rotación aleatoria
            rows, cols = imagen.shape
            for angulo in range(-20, 21, 5):  # Rotación en intervalos de 5 grados
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angulo, 1)
                imagen_rotada = cv2.warpAffine(imagen, M, (cols, rows))
                imagenes.append(imagen_rotada)
                etiquetas.append(nombre_persona)

            # Desenfoque gaussiano
            imagen_desenfocada = cv2.GaussianBlur(imagen, (5, 5), 0)
            imagenes.append(imagen_desenfocada)
            etiquetas.append(nombre_persona)

            # Ruido gaussiano
            mean = 0
            var = 0.1 * 255
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, imagen.shape)
            imagen_con_ruido = imagen + gaussian
            imagenes.append(imagen_con_ruido)
            etiquetas.append(nombre_persona)

    imagenes.append(np.zeros((100, 100), dtype=np.uint8))  # Imagen en blanco para representar "Desconocido"
    etiquetas.append("Desconocido")

    # Ruta donde se guardará el archivo etiquetas.npy
    ruta_etiquetas = 'etiquetas.npy'

    # Guardar las etiquetas en un archivo .npy
    np.save(ruta_etiquetas, etiquetas)

    return imagenes, etiquetas

def entrenamiento(request):
    # Ruta del dataset
    ruta_dataset = 'dataset/'

    # Cargar imágenes y etiquetas del dataset con aumento de datos
    imagenes, etiquetas = cargar_imagenes_y_etiquetas(ruta_dataset)

    # Convertir listas a arrays numpy
    imagenes = np.array(imagenes)
    etiquetas = np.array(etiquetas)

    # Normalizar las imágenes
    imagenes = imagenes.astype('float32') / 255.0

    # Entrenar el clasificador KNN
    knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    knn_clf.fit(imagenes.reshape(len(imagenes), -1), etiquetas)

    # Guardar el modelo KNN entrenado con joblib
    joblib.dump(knn_clf, 'modelo_knn_con_aumento_con_desconocido.pkl')

    # Entrenar el clasificador SVM
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_clf.fit(imagenes.reshape(len(imagenes), -1), etiquetas)

    # Guardar el modelo SVM entrenado con joblib
    joblib.dump(svm_clf, 'modelo_svm_con_aumento_con_desconocido.pkl')

    return HttpResponse("success")

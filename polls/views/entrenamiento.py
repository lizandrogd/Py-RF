import face_recognition
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
            imagen = face_recognition.load_image_file(imagen_ruta, mode='RGB')
            caras_codificadas = face_recognition.face_encodings(imagen, num_jitters=1)

            if len(caras_codificadas) > 0:
                imagen_codificada = caras_codificadas[0]
                imagenes.append(imagen_codificada)
                etiquetas.append(nombre_persona)
                
                # Aumento de datos
                imagen_flip = np.fliplr(imagen)
                caras_codificadas_flip = face_recognition.face_encodings(imagen_flip)
                if len(caras_codificadas_flip) > 0:
                    imagenes.append(caras_codificadas_flip[0])
                    etiquetas.append(nombre_persona)

                for angulo in range(-20, 21, 5):
                    M = cv2.getRotationMatrix2D((imagen.shape[1] / 2, imagen.shape[0] / 2), angulo, 1)
                    imagen_rotada = cv2.warpAffine(imagen, M, (imagen.shape[1], imagen.shape[0]))
                    caras_codificadas_rotadas = face_recognition.face_encodings(imagen_rotada)
                    if len(caras_codificadas_rotadas) > 0:
                        imagenes.append(caras_codificadas_rotadas[0])
                        etiquetas.append(nombre_persona)

                imagen_desenfocada = cv2.GaussianBlur(imagen, (5, 5), 0)
                caras_codificadas_desenfocadas = face_recognition.face_encodings(imagen_desenfocada)
                if len(caras_codificadas_desenfocadas) > 0:
                    imagenes.append(caras_codificadas_desenfocadas[0])
                    etiquetas.append(nombre_persona)

                mean = 0
                var = 0.1 * 255
                sigma = var ** 0.5
                gaussian = np.random.normal(mean, sigma, imagen.shape)
                imagen_con_ruido = imagen + gaussian
                caras_codificadas_con_ruido = face_recognition.face_encodings(imagen_con_ruido)
                if len(caras_codificadas_con_ruido) > 0:
                    imagenes.append(caras_codificadas_con_ruido[0])
                    etiquetas.append(nombre_persona)
            else:
                print(f"No se detectaron caras en la imagen: {imagen_ruta}")

    # Añadir vector de ceros para "Desconocido"
    imagenes.append(np.zeros(128))  # 128 es la longitud del vector de codificación de `face-recognition`
    etiquetas.append("Desconocido")

    # Guardar etiquetas
    np.save('etiquetas.npy', etiquetas)

    return np.array(imagenes), np.array(etiquetas)

def entrenamiento(request):
    ruta_dataset = 'dataset/'
    try:
        imagenes, etiquetas = cargar_imagenes_y_etiquetas(ruta_dataset)
    except Exception as e:
        return HttpResponse(f"Error al cargar imágenes y etiquetas: {str(e)}")

    if len(imagenes) < 2:
        return HttpResponse("Error: No hay suficientes imágenes para entrenar.")

    X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)

    # Entrenamiento y evaluación del clasificador KNN
    knn_clf = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
    try:
        knn_clf.fit(X_train, y_train)
        knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=5)
        knn_accuracy = accuracy_score(y_test, knn_clf.predict(X_test))
    except Exception as e:
        return HttpResponse(f"Error durante el entrenamiento KNN: {str(e)}")

    # Guardar modelo KNN
    try:
        joblib.dump(knn_clf, 'modelo_knn_con_aumento_con_desconocido.pkl')
    except Exception as e:
        return HttpResponse(f"Error al guardar modelo KNN: {str(e)}")

    # Entrenamiento y evaluación del clasificador SVM
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    try:
        svm_clf.fit(X_train, y_train)
        svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=5)
        svm_accuracy = accuracy_score(y_test, svm_clf.predict(X_test))
    except Exception as e:
        return HttpResponse(f"Error durante el entrenamiento SVM: {str(e)}")

    # Guardar modelo SVM
    try:
        joblib.dump(svm_clf, 'modelo_svm_con_aumento_con_desconocido.pkl')
    except Exception as e:
        return HttpResponse(f"Error al guardar modelo SVM: {str(e)}")

    num_imagenes = len(imagenes)
    num_perfiles = len(set(etiquetas))

    response = (
        f"Entrenamiento completado.<br>"
        f"KNN Accuracy: {knn_accuracy:.2f}<br>"
        f"SVM Accuracy: {svm_accuracy:.2f}<br>"
        f"Número de imágenes entrenadas: {num_imagenes}<br>"
        f"Número de perfiles entrenados: {num_perfiles}"
    )

    return HttpResponse(response)

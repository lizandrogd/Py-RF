import cv2
import numpy as np

# Ruta donde se encuentra el modelo entrenado
ruta_modelo_entrenado = "modelo_entrenado.xml"

# Ruta donde se encuentran los datos de entrenamiento
ruta_datos_entrenamiento = "datos_entrenamiento.txt"

# Cargar el modelo de reconocimiento facial LBPH
modelo_reconocimiento = cv2.face.LBPHFaceRecognizer_create()
modelo_reconocimiento.read(ruta_modelo_entrenado)

def cargar_datos_entrenamiento(ruta_datos_entrenamiento):
    """
    Carga los datos de entrenamiento desde un archivo de texto.
    """
    with open(ruta_datos_entrenamiento, "r") as f:
        datos_entrenamiento = f.read()
    return datos_entrenamiento

def reconocimiento_facial(modelo, ruta_datos_entrenamiento):
    """
    Realiza el reconocimiento facial en tiempo real.
    """
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    # Cargar los datos de entrenamiento
    datos_entrenamiento = cargar_datos_entrenamiento(ruta_datos_entrenamiento)

    while True:
        # Capturar un fotograma de la cámara
        ret, frame = cap.read()

        # Convertir la imagen a escala de grises
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Realizar el reconocimiento facial
        etiqueta, confianza = modelo.predict(frame_gris)

        # Mostrar el nombre del perfil y la confianza
        datos_reconocimiento = f'Perfil ID: {etiqueta}, Confianza: {confianza:.2f}'
        cv2.putText(frame, datos_reconocimiento, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Mostrar los datos de entrenamiento solo si la confianza es suficiente
        if confianza < 70:  # Ajusta este umbral según sea necesario
            cv2.putText(frame, datos_entrenamiento, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostrar el fotograma
        cv2.imshow('Reconocimiento Facial', frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reconocimiento_facial(modelo_reconocimiento, ruta_datos_entrenamiento)
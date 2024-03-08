import cv2
import numpy as np
import pymongo

def detect_face(frame, face_cascade):
    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta caras en la imagen en escala de grises
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces

def main():
    # Carga el clasificador Haar para detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

 # Establece la conexión a la base de datos MongoDB
    client = pymongo.MongoClient("mongodb://lizandrogd:Nicolas2796*+@195.35.32.59:27017/")  # Cambia localhost y el puerto si es necesario
    db = client["facialcheck"]  # Cambia "mydatabase" por el nombre de tu base de datos
    collection = db["imagenes"]  # Cambia "userdata" por el nombre de tu colección

    # Captura una imagen de referencia
    reference_image = cv2.imread("C:\laragon\www\facialcheck\storage\app\storage")  # Reemplaza "tu_imagen_de_referencia.jpg" con la ruta de tu imagen de referencia
    if reference_image is None:
        print("No se pudo cargar la imagen de referencia.")
        return

    # Convierte la imagen de referencia a escala de grises
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Crea el objeto de reconocimiento facial LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Entrena el reconocedor facial con la imagen de referencia
    recognizer.train([reference_gray], np.array([0]))

    # Abre la cámara
    cap = cv2.VideoCapture(0)

    # Verifica si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    # Lee y muestra imágenes de la cámara hasta que se presione la tecla 'q'
    while True:
        ret, frame = cap.read()  # Lee un fotograma de la cámara
        if not ret:
            print("Error al leer el fotograma de la cámara")
            break

        # Detecta caras en el fotograma
        faces = detect_face(frame, face_cascade)

        # Compara cada cara detectada con la imagen de referencia
        for (x, y, w, h) in faces:
            face_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            label, confidence = recognizer.predict(face_gray)

            # Si la diferencia entre las imágenes es menor que un umbral, se considera una coincidencia
            if confidence < 50:
                cv2.putText(frame, "Usuario identificado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # Aquí puedes mostrar los datos del usuario identificado
            else:
                cv2.putText(frame, "Usuario no identificado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Camera", frame)  # Muestra el fotograma en una ventana llamada "Camera"

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Espera hasta que se presione la tecla 'q'
            break

    # Libera la cámara y cierra todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
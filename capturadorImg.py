import cv2
import os

def capturar_rostro(nombre_persona, face_cascade):
    # Carpeta para almacenar los rostros
    carpeta = 'dataset/' + nombre_persona

    # Crear la carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    # Iniciar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)

    # Tamaño de la región de interés (ROI)
    ROI_SIZE = 350

    # Loop para capturar una sola imagen de la cámara
    while True:
        # Leer un frame de la cámara
        ret, frame = cap.read()

        # Convertir el frame a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Si se detecta al menos un rostro, capturar la primera imagen y salir del bucle
        if len(faces) > 0:
            # Verificar si el rostro está centrado en el cuadro de la cámara
            (x, y, w, h) = faces[0]
            centro_x = x + w // 2
            centro_y = y + h // 2
            if (centro_x > frame.shape[1] // 3) and (centro_x < 2 * frame.shape[1] // 3):
                # Dibujar un rectángulo azul como guía de referencia
                cv2.rectangle(frame, (centro_x - ROI_SIZE // 2, centro_y - ROI_SIZE // 2), (centro_x + ROI_SIZE // 2, centro_y + ROI_SIZE // 2), (255, 0, 0), 2)
                
                # Mostrar mensaje de instrucciones
                cv2.putText(frame, "Presione 'q' para tomar la foto", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Mostrar el frame con los rostros detectados y la guía de referencia
                cv2.imshow('Captura de Rostro', frame)

                # Guardar la imagen capturada si se presiona la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Asegurar que el tamaño de la ROI no exceda los límites del frame
                    x_roi = max(0, centro_x - ROI_SIZE // 2)
                    y_roi = max(0, centro_y - ROI_SIZE // 2)
                    roi_gray = gray[y_roi:y_roi+ROI_SIZE, x_roi:x_roi+ROI_SIZE]
                    
                    # Guardar la imagen capturada
                    cv2.imwrite(f"{carpeta}/rostro.jpg", roi_gray)
                    break
            else:
                # Mostrar un mensaje de error si el rostro no está centrado
                cv2.putText(frame, "Por favor, mire directamente a la cámara", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Mostrar un mensaje de error si no se detecta ningún rostro
            cv2.putText(frame, "No se detecta ningún rostro", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Mostrar el frame con los rostros detectados y la guía de referencia
        cv2.imshow('Captura de Rostro', frame)

        # Salir del loop si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura de video y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Cargar el clasificador de rostros en cascada de Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capturar el rostro de una persona específica
nombre_persona = input("Ingrese el nombre de la persona (sin espacios): ")
capturar_rostro(nombre_persona, face_cascade)

import cv2
from pymongo import MongoClient
from tkinter import Tk, filedialog
from bson import ObjectId  # Importar ObjectId desde bson
from datetime import datetime

# Conexión a la base de datos MongoDB
cliente = MongoClient('mongodb://localhost:27017')
base_datos = cliente['prueba']
coleccion_imagenes = base_datos['imagenes']

def seleccionar_imagen():
    # Abrir el explorador de archivos para seleccionar la imagen
    root = Tk()
    root.withdraw()  # Oculta la ventana principal
    ruta_imagen = filedialog.askopenfilename()  # Abre una ventana de diálogo para seleccionar la imagen
    root.destroy()  # Cierra la ventana después de seleccionar la imagen

    if not ruta_imagen:
        print("No se seleccionó ninguna imagen")
        return None

    # Leer el nombre de la imagen
    nombre_imagen = input("Ingrese el nombre de la imagen: ")

    # Leer el ID del perfil asociado a la imagen
    perfil_id = input("Ingrese el ID del perfil asociado a la imagen: ")

    # Obtener la fecha y hora actual
    fecha_actual = datetime.utcnow()

    # Leer la imagen desde la ruta especificada
    imagen = cv2.imread(ruta_imagen)

    if imagen is not None:
        # Convertir la imagen a formato binario
        _, img_encoded = cv2.imencode('.jpeg', imagen)

        # Insertar la imagen en la colección de imágenes
        imagen_id = coleccion_imagenes.insert_one({
            'nombre': nombre_imagen,
            'imagen': img_encoded.tobytes(),  # Guardar la imagen en formato binario
            'perfil_id': perfil_id,
            'updated_at': fecha_actual,
            'created_at': fecha_actual
        }).inserted_id

        print("Imagen subida con ID:", imagen_id)

        return imagen_id
    else:
        print("No se pudo cargar la imagen desde la ruta: ", ruta_imagen)
        return None

if __name__ == "__main__":
    # Ejemplo de uso
    imagen_id = seleccionar_imagen()
    if imagen_id:
        print("ID de la imagen seleccionada:", imagen_id)
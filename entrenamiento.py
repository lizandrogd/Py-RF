import cv2
import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
from bson import ObjectId

# Conexión a la base de datos MongoDB
cliente = MongoClient('mongodb://localhost:27017')
base_datos = cliente['prueba']
coleccion_imagenes = base_datos['imagenes']
coleccion_perfiles = base_datos['perfiles']

# Ruta donde se guardará el modelo entrenado
ruta_modelo_entrenado = "modelo_entrenado.xml"
ruta_datos_entrenamiento = "datos_entrenamiento.txt"

# Cargar el clasificador preentrenado de detección de rostros
clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar el modelo de reconocimiento facial LBPH
modelo_reconocimiento = cv2.face.LBPHFaceRecognizer_create()
def cargar_imagenes_desde_bd():
    imagenes = []
    etiquetas = []
    etiqueta_perfil_map = {}  # Mapeo de etiquetas a perfiles

    for imagen in coleccion_imagenes.find():
        perfil_id = imagen.get('perfil_id')
        perfil = coleccion_perfiles.find_one({"_id": ObjectId(perfil_id)})
        if perfil:
            # Asignar una etiqueta única a cada perfil
            if perfil_id not in etiqueta_perfil_map:
                etiqueta_perfil_map[perfil_id] = len(etiqueta_perfil_map)
            etiqueta = etiqueta_perfil_map[perfil_id]

            # Decodificar la imagen binaria y convertirla a formato numpy array
            img_encoded = imagen['imagen']
            nparr = np.frombuffer(img_encoded, np.uint8)
            img_decoded = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            # Detectar rostros en la imagen
            rostros = clasificador_rostros.detectMultiScale(img_decoded, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(rostros) > 0:
                for (x, y, w, h) in rostros:
                    # Recortar el área del rostro
                    rostro = img_decoded[y:y+h, x:x+w]
                    imagenes.append(rostro)
                    etiquetas.append(etiqueta)

    return imagenes, etiquetas

def guardar_datos_entrenamiento(perfiles):
    with open(ruta_datos_entrenamiento, "w") as f:
        for perfil in perfiles:
            f.write(f"Perfil ID: {perfil['_id']}\n")
            f.write("Nombre: " + perfil.get("nombre", "Desconocido") + "\n")
            f.write("Cédula: " + perfil.get("cedula", "Desconocido") + "\n")
            f.write("Edad: " + perfil.get("edad", "Desconocida") + "\n")
            f.write("\n")

def entrenar_modelo():
    imagenes, etiquetas = cargar_imagenes_desde_bd()

    if len(imagenes) == 0:
        print("No se encontraron imágenes en la base de datos para entrenar el modelo.")
        return

    modelo_reconocimiento.train(imagenes, np.array(etiquetas, dtype=np.int32))
    modelo_reconocimiento.save(ruta_modelo_entrenado)
    print(f"Modelo entrenado con éxito y guardado como '{ruta_modelo_entrenado}'.")

    # Obtener información de perfiles para guardar en datos de entrenamiento
    perfiles = list(coleccion_perfiles.find())
    guardar_datos_entrenamiento(perfiles)

    print("Datos entrenados:")
    for perfil in perfiles:
        print(f"Perfil ID: {perfil['_id']}")
        print("Nombre:", perfil.get("nombre", "Desconocido"))
        print("Cédula:", perfil.get("cedula", "Desconocido"))
        print("Edad:", perfil.get("edad", "Desconocida"))
        print()

if __name__ == "__main__":
    entrenar_modelo()
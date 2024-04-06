from django.http import JsonResponse
from bson import ObjectId
from datetime import datetime
from polls.mongo_connection import conectar_mongodb, obtener_coleccion

# Datos de conexión a MongoDB
dsn = 'mongodb+srv://lizandrogd:Natalia1095@cluster0.0dt5vq0.mongodb.net/test?retryWrites=true&w=majority'

database_name = 'facialcheck'
nombre_coleccion = 'perfiles'
nombre_coleccion_log = 'log'  # Nombre de la colección de logs

# Conectar a MongoDB y obtener las colecciones
client = conectar_mongodb(dsn)
coleccion_perfiles = obtener_coleccion(client, database_name, nombre_coleccion)
coleccion_log = obtener_coleccion(client, database_name, nombre_coleccion_log)

def procesar_resultados(resultados):
    perfiles_encontrados = []

    # Obtener la lista de cédulas de los resultados
    cedulas = resultados

    # Iterar sobre cada cédula en los resultados
    for cedula in cedulas:
        # Buscar en la colección perfiles por la cédula
        perfil = coleccion_perfiles.find_one({"cedula": cedula})
        if perfil:
            # Convertir el ObjectId a str
            perfil['_id'] = str(perfil['_id'])
            perfiles_encontrados.append(perfil)
            # Agregar registro al log
            agregar_log(perfil)

    # Aquí puedes realizar cualquier procesamiento adicional de los perfiles encontrados
    return JsonResponse({"resultados": perfiles_encontrados})

def agregar_log(perfil):
    # Obtener detalles del perfil para el log
    nombre = perfil.get('nombre', '')
    cedula = perfil.get('cedula', '')
    descripcion = 'ha sido reconocido'
    fecha_actual = datetime.now()
    hora_actual = fecha_actual.strftime("%H:%M:%S")
    fecha_actual_str = fecha_actual.strftime("%Y-%m-%d")

    # Construir el registro del log
    registro_log = {
        "detalle": f"{nombre}, {cedula}, {descripcion}",
        "fecha": fecha_actual_str,
        "hora": hora_actual
    }

    # Insertar el registro en la colección de log
    coleccion_log.insert_one(registro_log)

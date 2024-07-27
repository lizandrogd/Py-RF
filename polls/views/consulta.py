from django.http import JsonResponse
from bson import ObjectId
from datetime import datetime
from polls.mongo_connection import conectar_mongodb, obtener_coleccion
import os
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
# Datos de conexión a MongoDB
dsn = 'mongodb+srv://lizandrogd:Natalia1095@cluster0.0dt5vq0.mongodb.net/test?retryWrites=true&w=majority'

database_name = 'facialcheck'
nombre_coleccion = 'perfiles'
nombre_coleccion_log = 'log'  # Nombre de la colección de logs
nombre_coleccion_desconocido = 'desconocido'  # Nombre de la colección de logs

# Credenciales y configuración del correo
SMTP_HOST = 'smtp.hostinger.com'
SMTP_PORT = 465
SMTP_USERNAME = 'app@u-site.app'
SMTP_PASSWORD = 'Colombia2024*+'
MAIL_FROM_ADDRESS = 'app@u-site.app'
MAIL_FROM_NAME = 'Facial check'

# Conectar a MongoDB y obtener las colecciones
client = conectar_mongodb(dsn)
coleccion_perfiles = obtener_coleccion(client, database_name, nombre_coleccion)
coleccion_log = obtener_coleccion(client, database_name, nombre_coleccion_log)
coleccion_desconocido = obtener_coleccion(client, database_name, nombre_coleccion_desconocido)

def procesar_resultados(resultados):
    perfiles_rostro = []

    for cedula in resultados:
        if cedula != "Desconocido":
            # Buscar en la colección perfiles por la cédula
            print("Buscando perfil para la cédula:", cedula)
            perfil = coleccion_perfiles.find_one({"cedula": cedula})
            if perfil:
                # Convertir el ObjectId a str
                perfil['_id'] = str(perfil['_id'])
                perfiles_rostro.append(perfil)
                # Agregar registro al log
                print("Perfil encontrado:", perfil)
                agregar_log(perfil)
            else:
                print("Perfil no encontrado para la cédula:", cedula)
        else:
            perfiles_rostro.append({"cedula": "Desconocido"})

    # Aquí puedes realizar cualquier procesamiento adicional de los perfiles encontrados
    return JsonResponse({"resultados": perfiles_rostro, "consulta": resultados})


def agregar_log(perfil):
    # Obtener detalles del perfil para el log
    nombre = perfil.get('nombre', '')
    cedula = perfil.get('cedula', '')
    categoria = perfil.get('categoria_id', '')  

    descripcion = 'ha sido reconocido'

    if categoria == '669bc1f40500f8b6f80eabab':
        destinatario = 'admin@facialcheck.co'  # Cambia esto con el destinatario real
        asunto = 'Transgresor detectado'
        mensaje_html = '<p> El usuario  transgresor <b>HTML</b></p>',f"{nombre}, {cedula}, {descripcion}"
        enviar_correo(destinatario, asunto, mensaje_html)
    
    # Obtener la fecha y hora actual en el formato especificado
    fecha_actual = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'

    # Construir el registro del log
    registro_log = {
        "detalle": f"{nombre}, {cedula}, {descripcion}",
        "created_at": fecha_actual,
        "cedula": cedula,
    }

    # Insertar el registro en la colección de log
    coleccion_log.insert_one(registro_log)
    print("Registro de log agregado:", registro_log)

def guardar_rostro_desconocido(rostro):

    # Crear la ruta si no existe
    directorio_desconocidos = '/var/www/facialcheck/public/storage/desconocidos'
    if not os.path.exists(directorio_desconocidos):
        os.makedirs(directorio_desconocidos)

    # Generar un nombre único para la imagen
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
    nombre_archivo = f'desconocido_{timestamp}.jpg'
    ruta_archivo = os.path.join(directorio_desconocidos, nombre_archivo)

    # Guardar la imagen
    cv2.imwrite(ruta_archivo, rostro)

    # Obtener la fecha y hora actual en el formato especificado
    fecha_actual = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'

    # Construir el registro del log
    registro_desconocido = {
        "url": f"storage/desconocidos/{nombre_archivo}",
        "created_at": fecha_actual
    }

    # Insertar el registro en la colección de desconocido
    coleccion_desconocido.insert_one(registro_desconocido)
    print("Registro de log agregado:", registro_desconocido)


# Definir la función para enviar el correo
def enviar_correo(destinatario, asunto, mensaje_html):
    # Crear el objeto del mensaje
    msg = MIMEMultipart()
    msg['From'] = MAIL_FROM_ADDRESS
    msg['To'] = destinatario
    msg['Subject'] = asunto

    # Adjuntar el cuerpo del mensaje en HTML
    msg.attach(MIMEText(mensaje_html, 'html'))

    try:
        # Conectar al servidor SMTP
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        print('Correo enviado exitosamente')
    except Exception as e:
        print(f"No se pudo enviar el correo. Error: {e}")

# Verificar la condición y enviar el correo
categoria = '669bc1f40500f8b6f80eabab'  # Cambia esto con la lógica real de obtención de categoría

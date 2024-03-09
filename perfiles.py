from pymongo import MongoClient
from bson import ObjectId

# Conexión a la base de datos MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['prueba']  # Reemplaza 'tu_basedatos' por el nombre de tu base de datos
collection = db['perfiles']  # Nombre de la colección

def crear_perfil():
    nombre = input("Ingrese el nombre: ")
    cedula = input("Ingrese la cédula: ")
    edad = input("Ingrese la edad: ")
    descripcion = input("Ingrese la descripción: ")
    email = input("Ingrese el email: ")

    # Crear el documento
    perfil = {
        "nombre": nombre,
        "cedula": cedula,
        "edad": edad,
        "descripcion": descripcion,
        "email": email
    }

    # Insertar el perfil en la base de datos
    perfil_id = collection.insert_one(perfil).inserted_id
    print("Perfil creado correctamente con ID:", perfil_id)

def obtener_perfiles():
    perfiles = collection.find()
    for perfil in perfiles:
        print(perfil)

def actualizar_perfil():
    perfil_id = input("Ingrese el ID del perfil que desea actualizar: ")
    perfil = collection.find_one({'_id': ObjectId(perfil_id)})
    if perfil:
        print("Perfil encontrado:", perfil)
        nombre = input("Ingrese el nuevo nombre: ")
        cedula = input("Ingrese la nueva cédula: ")
        edad = input("Ingrese la nueva edad: ")
        descripcion = input("Ingrese la nueva descripción: ")
        email = input("Ingrese el nuevo email: ")

        # Actualizar el perfil en la base de datos
        result = collection.update_one({'_id': ObjectId(perfil_id)}, {'$set': {
            "nombre": nombre,
            "cedula": cedula,
            "edad": edad,
            "descripcion": descripcion,
            "email": email
        }})
        if result.modified_count > 0:
            print("Perfil actualizado correctamente")
        else:
            print("No se pudo actualizar el perfil")
    else:
        print("Perfil no encontrado")

def eliminar_perfil():
    perfil_id = input("Ingrese el ID del perfil que desea eliminar: ")
    result = collection.delete_one({'_id': ObjectId(perfil_id)})
    if result.deleted_count > 0:
        print("Perfil eliminado correctamente")
    else:
        print("Perfil no encontrado")

if __name__ == '__main__':
    while True:
        print("\nSelecciona una opción:")
        print("1. Crear perfil")
        print("2. Obtener perfiles")
        print("3. Actualizar perfil")
        print("4. Eliminar perfil")
        print("5. Salir")

        opcion = input("Ingrese el número de la opción: ")

        if opcion == "1":
            crear_perfil()
        elif opcion == "2":
            obtener_perfiles()
        elif opcion == "3":
            actualizar_perfil()
        elif opcion == "4":
            eliminar_perfil()
        elif opcion == "5":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Por favor ingrese un número del 1 al 5.")
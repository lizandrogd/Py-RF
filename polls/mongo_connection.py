import pymongo

def conectar_mongodb(dsn):
    """
    Función para establecer la conexión con la base de datos MongoDB.
    """
    client = pymongo.MongoClient(dsn)
    return client

def obtener_coleccion(client, database_name, nombre_coleccion):
    """
    Función para obtener una colección específica de la base de datos.
    """
    db = client[database_name]
    return db[nombre_coleccion]

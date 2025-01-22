from django.http import JsonResponse

def procesar_resultados(resultados):
    perfiles_rostro = []

    for cedula in resultados:
        if cedula == "Desconocido":
            perfiles_rostro.append({"cedula": cedula, "estado": "desconocido", "mensaje": "Rostro desconocido"})
        else:
            perfiles_rostro.append({"cedula": cedula, "estado": "reconocido", "mensaje": f"Rostro con cédula {cedula} reconocido exitosamente"})
        
    return JsonResponse({"resultados": perfiles_rostro, "consulta": resultados})

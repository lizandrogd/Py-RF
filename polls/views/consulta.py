from django.http import JsonResponse
import subprocess

def procesar_resultados(resultados):
    perfiles_rostro = []

    for cedula in resultados:
        if cedula == "Desconocido":
            perfiles_rostro.append({"cedula": cedula, "estado": "desconocido", "mensaje": "Rostro desconocido"})
        else:
            perfiles_rostro.append({"cedula": cedula, "estado": "reconocido", "mensaje": f"Rostro con cédula {cedula} reconocido exitosamente"})
        
    reiniciar_gunicorn
    return JsonResponse({"resultados": perfiles_rostro, "consulta": resultados})



def reiniciar_gunicorn():
    try:
        # Ejecutar el comando 'sudo systemctl restart gunicorn' usando subprocess
        subprocess.run(['sudo', 'systemctl', 'restart', 'gunicorn'], check=True)
        return 'Servidor reiniciado con éxito'
    except subprocess.CalledProcessError as e:
        # Si ocurre un error al ejecutar el comando
        return f'Error al reiniciar el servidor: {e}'

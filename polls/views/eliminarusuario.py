import os
import shutil
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes

@csrf_exempt
@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def eliminar_usuario(request):
    usuario = request.user
    documento = request.data.get("documento", "")
    
    if not documento:
        return JsonResponse({"error": True, "mensaje": "Debe proporcionar el número de documento."})
    
    carpeta = os.path.join('dataset', str(usuario), documento)
    
    if os.path.exists(carpeta):
        try:
            shutil.rmtree(carpeta)
            return JsonResponse({"error": False, "mensaje": "Usuario eliminado correctamente."})
        except Exception as e:
            return JsonResponse({"error": True, "mensaje": f"Error al eliminar la carpeta: {str(e)}"})
    else:
        return JsonResponse({"error": True, "mensaje": "El usuario no existe en la base de datos."})

import os
import numpy as np
import cv2
import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
import joblib
import face_recognition

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def reconocimiento_kyc(request):
    usuario = request.user
    imagen = request.FILES.get('image')
    documento = request.POST.get('documento')

    if not imagen:
        return JsonResponse({"error": True, "message": "No se proporcionó una imagen."})

    modelo_path = os.path.join('modelos_kyc', str(usuario), f"{documento}.pkl")

    if not os.path.exists(modelo_path):
        return JsonResponse({"error": True, "message": "No hay un perfil KYC registrado para este usuario."})

    try:
        # Procesar la imagen
        nparr = np.frombuffer(imagen.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JsonResponse({"error": True, "message": "Error al procesar la imagen."})

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)

        if not face_encodings:
            return JsonResponse({"error": True, "message": "No se detectó ningún rostro en la imagen."})

        # Cargar el embedding guardado del usuario
        embedding_registrado = joblib.load(modelo_path)

        # Comparar con la imagen ingresada
        distancia = np.linalg.norm(embedding_registrado - face_encodings[0])

        # Umbral de similitud
        UMBRAL_SIMILITUD = 0.6  
        es_reconocido = distancia < UMBRAL_SIMILITUD

        # Guardar la imagen si es reconocida
        if es_reconocido:
            ruta_guardado = os.path.join('registros_kyc', str(usuario), str(documento))
            os.makedirs(ruta_guardado, exist_ok=True)

            # Generar un nombre de archivo con la fecha y hora actual
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_imagen = os.path.join(ruta_guardado, f"{timestamp}.jpg")

            cv2.imwrite(ruta_imagen, img)  # Guardar la imagen

            return JsonResponse({
                "error": False,
                "message": "Reconocido",
                "imagen_guardada": ruta_imagen
            })

        return JsonResponse({"error": False, "message": "No reconocido"})

    except Exception as e:
        return JsonResponse({"error": True, "message": f"Error en la verificación: {e}"})

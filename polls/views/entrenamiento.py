import os
import numpy as np
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
import joblib
import face_recognition

logger = logging.getLogger(__name__)

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])  
def entrenamiento_kyc(request):
    usuario = request.user
    documento = request.POST.get('documento')

    dataset_path = os.path.join('dataset', str(usuario), str(documento))

    if not os.path.exists(dataset_path):
        return JsonResponse({"error": True, "message": f"No se encontró el dataset en {dataset_path}."})

    embeddings = []

    for filename in sorted(os.listdir(dataset_path)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(dataset_path, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if len(face_encodings) == 1:
                    embeddings.append(face_encodings[0])
                elif len(face_encodings) > 1:
                    logger.warning(f"Más de un rostro detectado en {image_path}. Ignorando esta imagen.")

            except Exception as e:
                logger.error(f"Error al procesar la imagen {image_path}: {e}")

    if len(embeddings) < 2:
        return JsonResponse({"error": True, "message": "Debe haber al menos dos imágenes válidas para generar el perfil."})

    # Calcular el embedding promedio del usuario
    promedio_embedding = np.mean(embeddings, axis=0)

    modelo_path = os.path.join('modelos_kyc', str(usuario))
    os.makedirs(modelo_path, exist_ok=True)

    # Guardar el embedding del usuario
    joblib.dump(promedio_embedding, os.path.join(modelo_path, f"{documento}.pkl"))

    return JsonResponse({
        "error": False,
        "message": "Perfil KYC registrado exitosamente.",
        "num_imagenes": len(embeddings),
    })

from django.urls import path
from polls.views.viewcapturadorImg import capturar_rostro
from polls.views.entrenamiento import entrenamiento
from polls.views.reconocimiento import reconocimiento_facial
from polls.views.eliminarusuario import eliminar_usuario

from polls.views import main
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenVerifyView

# Esta vista ya está incluida en SimpleJWT, solo la agregas a tus URLs
class CustomTokenObtainPairView(TokenObtainPairView):
    # Aquí puedes personalizar la respuesta si lo deseas
    pass

urlpatterns = [
    path('', main.index, name='index'),
    # Endpoints protegidos con JWT
    path('api/v1/capturar_rostro/', capturar_rostro, name="capturar_rostro"),
    path('api/v1/entrenamiento/', entrenamiento, name="entrenamiento"),
    path('api/v1/reconocimiento_facial/',  reconocimiento_facial, name="reconocimiento_facial"),
    path('api/v1/eliminar_usuario/', eliminar_usuario, name="eliminar_usuario"),


    # Endpoints para obtener y refrescar el token JWT
    path('api/v1/login/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/v1/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/v1/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
]

from django.urls import path
from polls.views.viewcapturadorImg import capturar_rostro_get
from polls.views.entrenamiento import entrenamiento
from polls.views.main import reconocimiento_facial


urlpatterns = [
    path("capturar_rostro", capturar_rostro_get, name="capturar_rostro"),
    path("entrenamiento", entrenamiento, name="entrenamiento"),
    path("reconocimiento_facial", reconocimiento_facial, name="reconocimiento_facial"),
]
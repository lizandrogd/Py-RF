from django.urls import path
from polls.views.viewcapturadorImg import capturar_rostro
from polls.views.entrenamiento import entrenamiento
from polls.views.reconocimiento import reconocimiento_facial
from polls.views import main


urlpatterns = [
    path('', main.index, name='index'),
    path("capturar_rostro", capturar_rostro, name="capturar_rostro"),
    path("entrenamiento", entrenamiento, name="entrenamiento"),
    path("reconocimiento_facial", reconocimiento_facial, name="reconocimiento_facial"),
]
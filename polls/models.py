from django.db import models

class Log(models.Model):
    # Definir los campos para el modelo Log
    level = models.CharField(max_length=50)  # Nivel de log (INFO, ERROR, etc.)
    message = models.TextField()  # El mensaje del log
    created_at = models.DateTimeField(auto_now_add=True)  # Fecha de creación

    def __str__(self):
        return f"[{self.level}] {self.message[:50]}"  # Muestra los primeros 50 caracteres del mensaje

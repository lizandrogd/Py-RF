from django.http import HttpResponse

def index(request):
    return HttpResponse("¡Este es un software de reconocimiento facial, creado por U-site!")

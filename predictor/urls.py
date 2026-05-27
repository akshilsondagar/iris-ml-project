from django.urls import path
from .views import predict_iris

urlpatterns = [
    path('', predict_iris, name='predict_iris'),
]
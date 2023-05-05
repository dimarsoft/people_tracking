from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
import redis
import requests
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, FileResponse, StreamingHttpResponse
import os

from django.views.generic import CreateView

from YoloApi.settings import BASE_DIR
import mimetypes
from rest_framework.response import Response

from main import get_version
from .models import VideoLoadingProcessing
from rest_framework import generics, status
from django.urls import reverse_lazy
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema


def video_view(request, filename):
    # Получаем путь к файлу по имени
    file_path = os.path.join(BASE_DIR, filename)
    # Проверяем, что файл существует
    if not os.path.exists(file_path):
        return HttpResponse(status=404)
    r = requests.get('http:///home/vladimir/PycharmProjects/Yolov8API/YoloApi/11_b4foC9D.mp4', stream=True)
    response = StreamingHttpResponse(streaming_content=r)
    response['Content-Disposition'] = f'attachement; filename="{filename}"'
    return response


from rest_framework import generics
from .serializers import VideoLoadingProcessingSerializer
from drf_yasg.utils import swagger_auto_schema


class VideoLoadingProcessingCreateView(generics.CreateAPIView):
    # queryset = VideoLoadingProcessing.objects.all()
    serializer_class = VideoLoadingProcessingSerializer

    @swagger_auto_schema(request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['file', 'email'],
        properties={
            'file': openapi.Schema(type=openapi.TYPE_FILE, description='The video file to process'),
            'email': openapi.Schema(type=openapi.TYPE_STRING, format='email',
                                    description='Email to send the result of video processing')
        },
    ))
    def post(self, request, *args, **kwargs):
        return super().post(request, *args, **kwargs)


redis_client = redis.Redis(host='localhost', port=6379, db=0)


class VideoProcessingInfoView(generics.RetrieveAPIView):
    queryset = VideoLoadingProcessing.objects.all()
    serializer_class = VideoLoadingProcessingSerializer

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.status == 'completed':
            data_result = instance.status
        else:
            data_result = redis_client.get(f'task_{instance.task_celery}')

        return Response({'result': data_result}, status=status.HTTP_200_OK)


class VideoProcessingListView(generics.ListAPIView):
    queryset = VideoLoadingProcessing.objects.all()
    serializer_class = VideoLoadingProcessingSerializer

    # @swagger_auto_schema(
    #     operation_summary="Список всех объектов VideoLoadingProcessing",
    #     responses={200: VideoLoadingProcessingSerializer(many=True)},
    # )
    # def list(self, request, *args, **kwargs):
    #     return super().list(request, *args, **kwargs)


class VideoProcessingDeleteView(generics.DestroyAPIView):
    queryset = VideoLoadingProcessing.objects.all()
    serializer_class = VideoLoadingProcessingSerializer


def index(request):
    return render(request, './index.html')


def about(request):
    soft_version = get_version
    context = {
        'soft_version': soft_version
    }

    return render(request, './about.html', context=context)


def videos(request):
    return render(request, "./videos.html")


# Создаем здесь представления.
def home(request):
    return render(request, "users/home.html")


class SignUp(CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy("login")
    template_name = "registration/signup.html"



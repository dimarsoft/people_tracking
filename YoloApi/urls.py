"""
URL configuration for YoloApi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from drf_yasg import openapi
from drf_yasg.views import get_schema_view

from YoloApi.rtsp import start_rtsp, stop_rtsp
# from users import

from app_detect import views
schema_view = get_schema_view(
    openapi.Info(
        title="API Documentation",
        default_version="v1",
        description="API documentation for YoloApi",
        # terms_of_service="https://www.google.com/policies/terms/",
        # contact=openapi.Contact(email="contact@myproject.com"),
        # license=openapi.License(name="BSD License"),
    ),
    public=True,
    # permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='home'),
    path('about', views.about, name='about'),
    path('video_info/<str:video_pk>', views.video_info, name='video_info'),
    path('video_info', views.video_info, name='video_info'),
    path('videos', views.videos, name='videos'),
    path('start_rtsp/', start_rtsp, name='start_rtsp'),
    path('stop_rtsp/', stop_rtsp, name='stop_rtsp'),

    path('videos/<str:filename>', views.video_view, name='video_view'),
    path('video-loading-processing/', views.VideoLoadingProcessingCreateView.as_view(),
         name='video_loading_processing_create'),
    path('get_video_processing_info/<int:pk>/', views.VideoProcessingInfoView.as_view(), name='video_processing_info'),
    path('video_processing_list/', views.VideoProcessingListView.as_view(), name='video_processing_list'),
    path('video_processing_delete/<int:pk>/', views.VideoProcessingDeleteView.as_view(), name='video_processing_delete'),
    path('api/documentation/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


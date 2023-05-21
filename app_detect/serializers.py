from rest_framework import serializers
from .models import VideoLoadingProcessing


class VideoLoadingProcessingSerializer(serializers.ModelSerializer):
    url_detail = serializers.HyperlinkedIdentityField(view_name='video_processing_info',
                                                      format='html')

    class Meta:
        model = VideoLoadingProcessing
        fields = ('id', 'email', 'file', 'url_detail')
        extra_kwargs = {
            'file': {'write_only': True},
        }

from django import forms
from .models import VideoLoadingProcessing


class FileUploadForm(forms.ModelForm):
    class Meta:
        model = VideoLoadingProcessing
        fields = ["email", 'file']

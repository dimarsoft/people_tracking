from django.contrib import admin


# Register your models here.
from .models import VideoLoadingProcessing


class VideoLoadingAdmin(admin.ModelAdmin):
    list_display = ('email', 'name')
    fields = ('email', 'file')


admin.site.register(VideoLoadingProcessing, VideoLoadingAdmin)
admin.site.site_header = 'Yolo Panel administration'
# admin.site.header = 'Yolo Panel'
admin.site.site_title = "Yolo Admin Portal"
admin.site.index_title = "Welcome to Yolo Portal"

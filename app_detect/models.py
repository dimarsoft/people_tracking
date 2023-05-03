import os
import time

from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver

from YoloApi import settings
from app_detect.detection_utils import predict_model


# Create your models here.
class VideoLoadingProcessing(models.Model):
    email = models.EmailField()
    file = models.FileField(upload_to='videos/')
    task_celery = models.CharField(max_length=100, blank=True, null=True)
    status = models.CharField(max_length=50, blank=True, null=True)

    @property
    def name(self):
        return self.file.name

    def __str__(self):
        return f"Video {self.id}: {self.file.name} ({self.status})"


class Task(models.Model):
    video = models.ForeignKey(VideoLoadingProcessing, on_delete=models.CASCADE, related_name='task_video')
    fps = models.FloatField(blank=True, null=True)
    counter_in = models.IntegerField(blank=True, null=True)
    counter_out = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return f"Task {self.id}: Video {self.video.file.name}"


class Deviations(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='task_deviations')
    start_frame = models.IntegerField()
    end_frame = models.IntegerField()
    code_deviations = models.CharField(max_length=50)

    def __str__(self):
        return f"Deviations {self.id}: Task {self.task.id} ({self.start_frame}-{self.end_frame})"


from YoloApi.tasks import video_processing_task

from celery.result import AsyncResult


@receiver(post_save, sender=VideoLoadingProcessing)
def video_processing(sender, instance, created, **kwargs):
    if created:

        path_file = os.path.join(settings.MEDIA_ROOT, instance.file.name)

        print(f"path_file = {path_file}")
        task_kwargs = dict()
        task_kwargs['source'] = path_file
        task_kwargs['save'] = True
        task_kwargs['classes'] = 0
        task_kwargs['line_thickness'] = 1
        task_kwargs['show'] = False
        # result = video_processing_task.apply_async(kwargs=task_kwargs)
        instance.status = 'processing'
        # instance.task_celery = result.id
        instance.save()




@receiver(post_save, sender=Task)
def get_save_post_for_reaction(sender, instance, created, **kwargs):
    if instance.status == 'Done':
        print('Done')

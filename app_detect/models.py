import os

from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver

from YoloApi import settings
from app_detect.detection_utils import predict_model


# Create your models here.
class VideoLoadingProcessing(models.Model):
    email = models.EmailField()
    file = models.FileField(upload_to='videos/')

    @property
    def name(self):
        return self.file.name

    def __str__(self):
        return f"Video {self.id}: {self.file.name}"


class Task(models.Model):
    video = models.ForeignKey(VideoLoadingProcessing, on_delete=models.CASCADE, related_name='task_video')
    task_celery = models.CharField(max_length=50)
    status = models.CharField(max_length=20)
    fps = models.FloatField(blank=True, null=True)
    counter_in = models.IntegerField(blank=True, null=True)
    counter_out = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return f"Task {self.id}: Video {self.video.file.name} ({self.status})"


class Deviations(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='task_deviations')
    start_frame = models.IntegerField()
    end_frame = models.IntegerField()
    code_deviations = models.CharField(max_length=50)

    def __str__(self):
        return f"Deviations {self.id}: Task {self.task.id} ({self.start_frame}-{self.end_frame})"


@receiver(post_save, sender=VideoLoadingProcessing)
def get_save_post_for_reaction(sender, instance, created, **kwargs):
    if created:
        path = os.path.join(settings.MEDIA_ROOT, instance.file.name)
        print(path)
        predict_model(source=path, save=True, classes=0, line_thickness=1, show=False, project=settings.BASE_DIR)


@receiver(post_save, sender=Task)
def get_save_post_for_reaction(sender, instance, created, **kwargs):
    if instance.status == 'Done':
        print('Done')

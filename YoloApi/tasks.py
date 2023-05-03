import time

import redis
from celery import shared_task
import logging

from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode

from YoloApi import settings
from YoloApi.celery import app

from celery.utils.log import get_task_logger
from django.core.mail import EmailMultiAlternatives, send_mail
from django.template.loader import render_to_string
from django.utils import timezone
import pytz

from app_detect.detection_utils import predict_model


@app.task(name="video_processing_task", bind=True)
def video_processing_task(self, *args, source=None, save=True, classes=0, line_thickness=1, show=False,
                          project=settings.BASE_DIR, **kwargs):
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    print(args, kwargs)
    task_id = self.request.id
    print(f"Task ID: {task_id}")
    #predict_model(source=source, save=save, classes=classes, line_thickness=line_thickness, show=show, project=project,
    #              task_id=task_id)
    redis_client.delete(f'task_{task_id}')

    return 'Task completed'

    # predict_model(source=source, save=save, classes=classes, line_thickness=line_thickness, show=show, project=project)

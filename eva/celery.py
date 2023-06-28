import os
import logging
from celery import Celery
from celery.signals import after_setup_logger, after_setup_task_logger
from celery.app.log import TaskFormatter

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eva.settings')

app = Celery('eva')

app.config_from_object('django.conf:settings', namespace='CELERY')


@after_setup_task_logger.connect
def setup_task_loggers(logger, *args, **kwargs):
    for handler in logger.handlers:
        handler.setFormatter(TaskFormatter(
            '[%(asctime)s] [%(task_name)s:%(task_id)s] %(message)s'
        ))


@after_setup_logger.connect
def setup_loggers(logger, *args, **kwargs):
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))


app.autodiscover_tasks()

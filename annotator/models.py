import os
import random
import colorsys
import json
import logging
from django.db import models
from django.contrib.staticfiles import finders
from django.db.models.signals import post_delete
from django.dispatch import receiver

logger = logging.getLogger(__name__)

def random_color():
    hue, sat, light = random.random(), 0.5 + random.random() / \
        2.0, 0.4 + random.random() / 5.0
    r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(hue, light, sat)]
    hex_code = '%02x%02x%02x' % (r, g, b)
    return hex_code


class Label(models.Model):
    """The classes available for workers to choose from for each object."""
    name = models.CharField(blank=True, max_length=100, unique=True,
                            help_text="Name of class label option.")
    color = models.CharField(max_length=6, default=random_color)

    def __str__(self):
        return self.name


class Project(models.Model):
    """The classes available for workers to choose from for each object."""
    name = models.CharField(blank=True, max_length=100, unique=True,
                            help_text="Name of class label option.")
    labels = models.ManyToManyField(Label, blank=True, through="LabelMapping")
    desc = models.CharField(blank=True, max_length=100, unique=False,
                            help_text="Name of class label option.")

    def __str__(self):
        return self.name


class LabelMapping(models.Model):
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    num = models.IntegerField()


class Video(models.Model):
    annotation = models.TextField(
        blank=True,
        help_text="A JSON blob containing all user annotation sent from client."
    )
    name = models.CharField(max_length=255, unique=True,
                            help_text="Name of the sequence.")
    date = models.DateTimeField(max_length=100, auto_now=True,
                                help_text="Date of when the video was added")
    label = models.ManyToManyField(Label, blank=True)
    cache_file = models.CharField(blank=True, max_length=200)
    cache_task_id = models.CharField(blank=True, max_length=36)
    extract_task_id = models.CharField(blank=True, max_length=36)
    project = models.ForeignKey(
        Project, on_delete=models.SET_NULL, blank=True, null=True)
    zipfile = models.FileField(upload_to='zipfiles', blank=True)
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    channels = models.IntegerField(default=0)

    @property
    def image_list(self):
        files = self.uploadfile_set.filter(file_type=UploadFile.IMAGE)
        images_info = sorted([(x.file.url, x.width, x.height) for x in files])
        return images_info

    def __str__(self):
        return '/video/{}'.format(self.id)
    
    @property
    def images(self):
        files = self.uploadfile_set.filter(file_type=UploadFile.IMAGE)
        files = sorted([(x.file.name, x.width, x.height) for x in files])
        return files
        
    @property
    def video(self):
        files = self.uploadfile_set.filter(file_type=UploadFile.VIDEO)
        return files

def unique_path(instance, filename):
    return os.path.join(str(instance.video.id), filename)

class UploadFile(models.Model):
    IMAGE = 'image'
    VIDEO = 'video'
    FILE_TYPES = ((IMAGE, 'Image'),
                  (VIDEO, 'Video'))
    file = models.FileField(upload_to=unique_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    video = models.ForeignKey('Video', on_delete=models.CASCADE, null=True)
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    file_type = models.CharField(
        max_length=5, choices=FILE_TYPES, default=IMAGE)

@receiver(post_delete, sender=UploadFile)
def image_delete_handler(sender, instance, **kwargs):
    if instance.file:
        logger.debug('Deleting {}'.format(instance.file.path))
        try:
            os.remove(instance.file.path)
        except FileNotFoundError:
            pass
        # remove the directory if this is the last file to delete
        try:
            os.rmdir(os.path.dirname(instance.file.path))
        except OSError:
            pass

# delete the image list, if the video object ie deleted from db
@receiver(post_delete, sender=Video)
def image_post_delete_handler(sender, instance, **kwargs):
    if instance.cache_file:
        logger.debug('Deleting {}'.format(instance.cache_file))
        try:
            os.remove(instance.cache_file)
        except FileNotFoundError:
            pass
    if instance.zipfile:
        logger.debug('Deleting {}'.format(instance.zipfile.path))
        try:
            os.remove(instance.zipfile.path)
        except FileNotFoundError:
            pass

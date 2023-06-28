import logging

from django.contrib import admin
from django.contrib.admin import SimpleListFilter
from .models import Video, Label, UploadFile
from django.db.models import Count, Sum, Q, Case, When, IntegerField

logger = logging.getLogger()


class VideoAdmin(admin.ModelAdmin):
    def video_url(self, obj):
        return '<a target="_" href="/video/{}/">/video/{}/</a>'.format(obj.id, obj.id)
    video_url.allow_tags = True
    video_url.short_description = 'Video'


admin.site.register(Video, VideoAdmin)
admin.site.register(Label)
admin.site.register(UploadFile)

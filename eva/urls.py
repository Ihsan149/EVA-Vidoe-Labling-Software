from django.conf.urls import url
from django.conf.urls.static import static
from django.conf import settings
from django.contrib import admin
from django.contrib.auth.views import login, logout
from django.views.generic.base import RedirectView
from django.views.static import serve

from annotator.views import *
from annotator.services import *

admin.site.site_header = 'eva'

urlpatterns = [
    url(r'^$', home),
    url(r'^favicon\.ico$', RedirectView.as_view(url='/static/img/icon.png')),
    url(r'^projects/$', projects, name='define_projects'),
    url(r'^video/(\d+)/(\d+)/$', VideoView.as_view(), name='video'),
    url(r'^annotation/(\d+)/$', AnnotationView.as_view()),
    url(r'^login/$', login,
        {'template_name': 'admin/login.html',
            'extra_context': {'site_header': 'BeaverDam Login'}
        }, name='login'),
    url(r'^logout/$', logout),
    url(r'^accounts/', RedirectView.as_view(url='/')),
    url(r'^admin/', admin.site.urls),
    url(r'^upload/(?P<name>.*)/done/$', UploadVideoDone.as_view(), name='upload_video_done'),
    url(r'^upload/(?P<name>.*)/$', UploadVideos.as_view(), name='upload_videos'),
    url(r'^create_video/(?P<name>.*)/$', CreateVideo.as_view(), name='create_video'),
    url(r'^tracker/(\d+)/$', tracker, name='tracker'),
    url(r'^tracker/get_results/$', tracker_get_results, name='tracker_get_results'),
    url(r'^check_video_name/(?P<name>.*)/$', check_video_name),
    url(r'^export/labels/(?P<name>.*)/$', ExportLabels.as_view(), name='export_labels'),
    url(r'^export/video/(?P<vid>\d+)/$', ExportVideo.as_view(), name='export_video'),
    url(r'^export/video/status/$', ExportVideoStatus.as_view()),
    url(r'^labels/labels/$', LabelsView.as_view(), name='labels'),
    url(r'^labels/project/$', ProjectView.as_view(), name='project'),
    url(r'^labels/label_select/$', LabelSelect.as_view(), name='label_select'),
    url(r'^labels/project_select/$', ProjectSelect.as_view(), name='project_select'),
    url(r'^labels/$', labels, name='labels'),
    url(r'^videos/$', Videos.as_view(), name='videos'),
    url(r'^video/status/(\d+)/$', VideoStatus.as_view())
]

# Add media folder
urlpatterns.append(url(
    r'^media/(?P<path>.*)$',
    serve,
    {'document_root': settings.MEDIA_ROOT, 'show_indexes': True}
))

# Add static when debug is off
if not settings.DEBUG:
    urlpatterns.append(url(
        r'^static/(?P<path>.*)$',
        serve,
        {'document_root': settings.STATIC_ROOT, 'show_indexes': True}
    ))

import io
import zipfile
import logging
import subprocess
import re
#import cv2
#import numpy as np
from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseBadRequest, HttpResponseNotFound
from django.views.generic import View
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.cache import never_cache
from django.db import IntegrityError
from django.db.models import Max
from django.http import JsonResponse
from celery.result import AsyncResult
from celery import chain
from celery import states as task_states
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .models import *
from .tasks import tracker_task, create_cache_task, convert_to_darknet, convert_to_pascal_voc, \
    extract_frames, VideoError, create_zipfile, clean_zipfiles
from .utils import *
import math


logger = logging.getLogger(__name__)


def home(request):
    return render(
        request, 'videolist.html',
        context={'projects': Project.objects.all()}
    )


def projects(request):
    return render(request, 'projects.html')


def labels(request):
    return render(request, 'labels.html')


class VideoView(View):
    video_chunk_size = settings.VIDEO_CHUNK_SIZE
    minimum_final_chunk = settings.MIN_FINAL_CHUNK_SIZE

    def validate_index(self, video_index, length_image_list):
        r = length_image_list / self.video_chunk_size
        if r < 1:
            max_value_of_index = 1
        else:
            max_value_of_index = math.ceil(r)
            if max_value_of_index - r > self.minimum_final_chunk / self.video_chunk_size:
                max_value_of_index -= 1
        if video_index > max_value_of_index - 1:
            return False
        return True

    def get_video_slice(self, video_index, length_image_list):
        last = False
        first = False
        if video_index == 0:
            first = True
        start_of_slice = video_index * self.video_chunk_size
        end_of_slice = min((video_index + 1) * self.video_chunk_size, length_image_list)
        if length_image_list - end_of_slice < self.minimum_final_chunk:
            end_of_slice = length_image_list
            last = True

        return first, last, start_of_slice, end_of_slice

    @xframe_options_exempt
    def get(self, request, video_id, video_index):
        video_index = int(video_index)

        try:
            video = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            raise Http404(
                'No video with id "{}". Possible fixes:\n'
                '1) Download an up to date DB, see README.'
                ' \n2) Add this video to the DB via /admin'.format(video_id)
            )

        length_image_list = len(video.image_list)
        # raise 404 when user tries to access non existing video_index
        if not self.validate_index(video_index, length_image_list):
            return HttpResponseNotFound()
        # Data for Javascript
        video_data = json.dumps({
            'id': video.id,
            'location': 'Image List',
            'path': '',
            'is_image_sequence': True if video.image_list else False,
            'annotated': video.annotation != '',
        })
        label_data = []
        video_labels = video.project.labels.all()
        project_name = Project.objects.get(id=video.project_id)

        for v_label in video_labels:
            label_data.append({'name': v_label.name, 'color': v_label.color})

        first, last, start_of_slice, end_of_slice = self.get_video_slice(video_index, length_image_list)
        video_slice = slice(start_of_slice, end_of_slice)
        video_display_data = {'project':project_name, 'name':video.name, 'images_left':length_image_list-end_of_slice}
        video_chunk = []
        video_dimension = []
        for img in video.image_list[video_slice]:
            video_chunk.append(img[0])
            if img[1] and img[2]:
                video_dimension.append([img[1], img[2]])  # append width, height of each image
            else:
                video_dimension.append([video.width, video.height])  # video object still has width and height
        # Data for python templating
        response = render(request, 'video.html', context={
            'label_data': label_data,
            'video_data': video_data,
            'image_list_dimensions': video_dimension,
            'image_list': video_chunk if video.image_list else 0,
            'image_list_path': '',
            'help_embed': True,
            'offset': video_index * self.video_chunk_size,
            'first_video_index': first,
            'last_video_index': last,
            'video_index': video_index,
            'video_chunk_size': len(video_chunk),
            'video_info': video_display_data
        })
        response['X-Frame-Options'] = 'SAMEORIGIN'
        return response


class AnnotationView(View):

    def get(self, request, video_id):
        video = Video.objects.get(id=video_id)
        return HttpResponse(video.annotation, content_type='application/json')

    def post(self, request, video_id):
        data = json.loads(request.body.decode('utf-8'))
        video = Video.objects.get(id=video_id)
        video.annotation = json.dumps(data['annotation'])
        video.save()  # save table in db
        return HttpResponse('success')


class ExportLabels(View):
    def get(self, request, name):
        if not 'id' in request.GET:
            return HttpResponse(status=403)

        ids = json.loads(request.GET['id'])

        if not hasattr(self, name):
            return HttpResponse(status=403)

        response = getattr(self, name)(ids)

        return response

    def yolo(self, video_ids):
        b = io.BytesIO()
        zf = zipfile.ZipFile(b, 'w')
        single_video = len(video_ids) == 1

        with zipfile.ZipFile(b, 'w') as zf:
            if single_video:
                vid = Video.objects.get(id=video_ids[0])
                name = vid.name
                files = convert_to_darknet(vid)
                for file, text in files.items():
                    zf.writestr(file, text)
            else:
                name = 'multiple'
                videos = [v for v in Video.objects.filter(id__in=video_ids)]
                for vid in videos:
                    files = convert_to_darknet(vid)
                    for file, text in files.items():
                        zf.writestr(os.path.join(vid.name, file), text)

        response = HttpResponse(
            b.getvalue(),
            content_type='application/x-zip-compressed'
        )
        response['Content-Disposition'] = (
            'attachment; '
            'filename={}.zip'
        ).format(name)
        return response

    def pascal_voc(self, video_ids):
        b = io.BytesIO()
        zf = zipfile.ZipFile(b, 'w')
        single_video = len(video_ids) == 1

        with zipfile.ZipFile(b, 'w') as zf:
            if single_video:
                vid = Video.objects.get(id=video_ids[0])
                name = vid.name
                files = convert_to_pascal_voc(vid)
                for file, text in files.items():
                    zf.writestr(file, text)
            else:
                name = 'multiple'
                videos = [v for v in Video.objects.filter(id__in=video_ids)]
                for vid in videos:
                    files = convert_to_pascal_voc(vid)
                    for file, text in files.items():
                        zf.writestr(os.path.join(vid.name, file), text)

        response = HttpResponse(
            b.getvalue(),
            content_type='application/x-zip-compressed'
        )
        response['Content-Disposition'] = (
            'attachment; '
            'filename={}.zip'
        ).format(name)
        return response

class UploadVideos(View):
    def get(self, request, name):
        try:
            video = Video.objects.get(name=name)
        except Video.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Does not exist'})
        files = video.uploadfile_set.all()

        file_list = []
        for file in files:
            file_list.append({
                'name': os.path.split(file.file.name)[1],
                'url': file.file.url,
                'size': file.file.size,
                # 'deleteUrl': reverse('delete_file', args=(file.id,)),
                'deleteType': 'POST',
                'type': 'image/%s' % (settings.IMAGE_FORMAT)
            })

        response_data = {
            'files': file_list,
            'status': 'success',
            'message': ''
        }

        return JsonResponse(response_data)

    def post(self, request, name=None):
        error = None
        if not request.FILES:
            error = 'Must upload a file.'
            return JsonResponse({'error': error})

        try:
            video = Video.objects.get(name=name)
        except Video.DoesNotExist:
            error = 'Could not upload. Video "{}" does not exist.'.format(name)

        files = request.FILES.getlist('file')
        logger.info('Uploading {} files'.format(len(files)))

        img_size = get_img_size_from_buffer(files[0])
        video.height = img_size[0]
        video.width = img_size[1]
        video.channels = img_size[2]
        video.save()

        file_list = []
        for file in files:
            file_list.append({
                "name": file.name,
                "size": file.size,
                "type": file.content_type
            })

            if error:
                file_list[0]['error'] = error
            else:
                img_dimension = get_img_size_from_buffer(file)
                file_db = UploadFile.objects.create()
                file_db.video = video
                file_db.file = file
                file_db.width = img_dimension[1]
                file_db.height = img_dimension[0]
                if file.content_type in ['video/mp4', 'video/quicktime', 'video/avi']:
                    file_db.file_type = UploadFile.VIDEO
                    if not settings.FFMPEG_BIN:
                        file_list[0]['error'] = 'ffmpeg not installed'
                file_db.save()
        response_data = {'files': file_list}

        return JsonResponse(response_data)


class CreateVideo(View):
    "Create video from uploaded images."

    def post(self, request, name):
        status = 'success'

        try:
            project = Project.objects.get(id=request.POST.get('project', None))
        except Project.DoesNotExist:
            status = 'error'
        else:
            try:
                video = Video.objects.create(name=name)
                video.project = project
                video.save()
            except IntegrityError:
                status = 'error'

        return JsonResponse({'status': status})


class UploadVideoDone(View):
    def post(self, request, name):
        video = Video.objects.get(name=name)
        video_files = video.uploadfile_set.filter(file_type=UploadFile.VIDEO)
        if video_files:
            task = chain(extract_frames.s() | create_cache_task.s())(video.id)
            video.extract_task_id = task.parent.task_id
        else:
            task = create_cache_task.delay(video.id)

        video.cache_task_id = task.task_id
        video.save()
        return HttpResponse('')


def tracker(request, video_id):
    resp = {'status': 'ok'}
    try:
        x = float(request.POST.get('x', ''))
        y = float(request.POST.get('y', ''))
        w = float(request.POST.get('w', ''))
        h = float(request.POST.get('h', ''))

        time = int(request.POST.get('t', ''))
        bbox = (x, y, w, h)
        task = tracker_task.delay(video_id, time, bbox)
        resp['task_id'] = task.task_id
    except ValueError:
        resp['status'] = 'error'
        resp['text'] = 'Incorrect input'

    return JsonResponse(resp)


def tracker_get_results(request):
    task_id = request.GET.get('task_id')
    if task_id:
        async_result = AsyncResult(task_id)
        if async_result.ready():
            return JsonResponse({
                'finish': async_result.ready(),
                'results': async_result.get()
            })
    return JsonResponse({'finish': False})


@never_cache
def check_video_name(request, name):
    query = Video.objects.filter(name=name)
    available = not query
    return JsonResponse({"nameAvailable": available})


def gen_new_name(db, base_name):
    old_names = db.objects.filter(
        name__regex=r'^{} \(\d+\)'.format(base_name)).values_list('name')
    num = [int(re.match(r'^[\w\ ]+\((\d+)\)', x[0]).group(1))
           for x in old_names]
    num = max(num) + 1 if num else 1
    new_name = '{} ({})'.format(base_name, num)
    return new_name


@method_decorator(csrf_exempt, name='dispatch')
class LabelsView(View):
    def get(self, request):
        labels = Label.objects.all()
        resp = []
        for label in labels:
            resp.append({'name': label.name,
                         'color': label.color,
                         'id': label.id,
                         'text': label.name,
                         'value': label.id})
        return JsonResponse(resp, safe=False)

    def post(self, request):
        if 'action' in request.POST:
            if request.POST['action'] == 'new':
                new_name = gen_new_name(Label, 'New label')
                row = Label(name=new_name)
                try:
                    row.save()
                except IntegrityError:
                    return HttpResponse(
                        status=500,
                        reason=("New label could not be saved to database, "
                                "please try again later.")
                    )
            elif request.POST['action'] == 'delete':
                ids = list()  # to save ids which cannot be deleted
                for id in request.POST.getlist('id[]'):
                    if Video.objects.filter(project_id__labelmapping__label_id=id).exists():
                        ids.append(id)
                    else:
                        Label.objects.get(id=id).delete()
                if len(ids):
                    return HttpResponse(reason='Error: Label(s) in use.', status=400)
        else:
            pk = request.POST['pk']
            field = request.POST['name']
            value = request.POST['value']

            row = Label.objects.get(id=pk)
            setattr(row, field, value)
            try:
                row.save()
            except IntegrityError:
                return HttpResponse('Error: Name not unique.', status=400)
        return HttpResponse('')


@method_decorator(csrf_exempt, name='dispatch')
class ProjectView(View):
    def get(self, reqest):
        projects = Project.objects.all()
        resp = []
        for project in projects:
            mapping = LabelMapping.objects.filter(project=project)
            resp.append({
                'name': project.name,
                'desc': project.desc,
                'labels': [x[0] for x in mapping.values_list('label__name')],
                'id': project.id
            })
        return JsonResponse(resp, safe=False)

    def post(self, request):
        if 'action' in request.POST:
            if request.POST['action'] == 'new':
                new_name = gen_new_name(Project, 'New project')
                row = Project(name=new_name)
                try:
                    row.save()
                except IntegrityError:
                    return HttpResponse(
                        reason='Error: Name not unique.',
                        status=400
                    )
            elif request.POST['action'] == 'delete':
                ids = list()  # to save ids which cannot be deleted
                for id in request.POST.getlist('id[]'):
                    if Video.objects.filter(project_id=id).exists():
                        ids.append(id)
                    else:
                        Project.objects.get(id=id).delete()
                if len(ids):
                    return HttpResponse(reason='Error: Project(s) in use.', status=400)
        else:
            pk = request.POST['pk']
            field = request.POST['name']
            value = request.POST['value']
            row = Project.objects.get(id=pk)
            setattr(row, field, value)
            try:
                row.save()
            except IntegrityError:
                return HttpResponse('Error: Name not unique.', status=400)
        return HttpResponse('')


@method_decorator(csrf_exempt, name='dispatch')
class LabelSelect(View):
    def get(self, request):
        mapping = LabelMapping.objects.filter(
            project__id=request.GET['project_id'])
        resp = []
        for m in mapping:
            resp.append({
                'name': m.label.id,
                'id': m.id,
                'num': m.num
            })
        return JsonResponse(resp, safe=False)

    def post(self, request):
        resp = {'status': 'success'}
        if 'action' in request.POST:
            if request.POST['action'] == 'delete':
                for id in request.POST.getlist('id[]'):
                    # if the project is associated to a video, restrict user to delete the label
                    # this condition will hit on the 1st id
                    if Video.objects.filter(project_id__labelmapping__id=id).exists():
                        return HttpResponse(reason='Error: Project in use.', status=400)
                    LabelMapping.objects.get(id=id).delete()
        else:
            project_id = request.POST['project_id']
            if int(request.POST['pk']) == -1:
                if request.POST['name'] == 'num':
                    return HttpResponse(
                        status=400,
                        reason='Select label before setting the number.'
                    )
                project = Project.objects.get(id=project_id)
                num = LabelMapping.objects.filter(
                    project__id=project_id).aggregate(Max('num'))['num__max']
                num = num+1 if num is not None else 0
                mapping = LabelMapping(project=project, num=num)
                resp['status'] = 'new'
            else:
                # If the project is associated to a video, restrict user to replace
                # selected label name.
                # Allow users to change label id
                existing_label_id = LabelMapping.objects.filter(
                    id=request.POST['pk']).values('label_id')
                if (Video.objects.filter(
                        project_id__labelmapping__label_id=existing_label_id,
                        project__id=project_id).exists()
                        and request.POST['name'] == 'name'):
                    return HttpResponse(reason='Error: Project in use.', status=400)
                mapping = LabelMapping.objects.get(id=request.POST['pk'])

            if request.POST['name'] == 'name':
                if LabelMapping.objects.filter(
                        project__id=project_id, label__id=request.POST['value']).exists():
                    return HttpResponse('Error: Label exists.', status=400)
                mapping.label = Label.objects.get(id=request.POST['value'])
            elif request.POST['name'] == 'num':
                try:
                    int(request.POST['value'])
                except ValueError:
                    return HttpResponse('Error: Not a number ', status=400)
                if LabelMapping.objects.filter(
                        project__id=project_id, num=request.POST['value']).exists():
                    return HttpResponse(
                        'Error: Number already used.',
                        status=400
                    )
                mapping.num = request.POST['value']
            mapping.save()
            resp['data'] = {'name': mapping.label.id,
                            'id': mapping.id,
                            'num': mapping.num}

        return JsonResponse(resp, safe=False)


@method_decorator(csrf_exempt, name='dispatch')
class Videos(View):
    def get(self, request):
        videos = Video.objects.all()
        resp = []
        for video in videos:
            resp.append({
                'name': video.name,
                'date': video.date,
                'project': video.project.id if video.project else None,
                'annotation': True if video.annotation else False,
                'id': video.id
            })
        return JsonResponse(resp, safe=False)

    def post(self, request):
        if 'action' in request.POST:
            if request.POST['action'] == 'delete':
                for id in request.POST.getlist('id[]'):
                    Video.objects.get(id=id).delete()
        else:
            pk = request.POST['pk']
            field = request.POST['name']
            value = request.POST['value']
            if field != 'project':
                return HttpResponseBadRequest("Only modifiable field is project.")
            row = Video.objects.get(id=pk)
            project = Project.objects.get(id=value)
            setattr(row, field, project)
            row.annotation = ''
            row.save()
        return HttpResponse('')


class ProjectSelect(View):
    def get(self, reqest):
        project = Project.objects.all()
        resp = []
        for project in project:
            resp.append({'value': project.id,
                         'text': project.name})
        return JsonResponse(resp, safe=False)


def is_task_running(task_id):
    res = AsyncResult(task_id).status == task_states.STARTED
    logger.debug('Task running: {}'.format(res))
    return res


def is_task_pending(task_id):
    res = AsyncResult(task_id).status == task_states.PENDING
    logger.debug('Task pending: {}'.format(res))
    return res


def is_task_done(task_id):
    res = AsyncResult(task_id).ready()
    logger.debug('Task done: {}'.format(res))
    return res


def get_task_result(task_id):
    task = AsyncResult(task_id)
    res = task.result if task.ready else None
    logger.debug('Task result: {}'.format(res))
    return res


def task_failed(task_id):
    res = AsyncResult(task_id).failed()
    logger.debug('Task failed: {}'.format(res))
    return res


class VideoStatus(View):
    def status(self, id):
        resp = {'status': 'ok', 'code': 0}
        try:
            video = Video.objects.get(id=id)
        except Video.DoesNotExist:
            resp = {'status': 'error',
                    'text': 'Video does not exists', 'code': 1}
        else:
            if not video.uploadfile_set.filter(file_type='image').exists():
                text = 'No images available'
                resp = {'status': 'error', 'text': text, 'code': 6}
                has_video = video.uploadfile_set.filter(file_type='video').exists()
                if has_video and settings.FFMPEG_BIN:
                    text = text + '. Extracting frames...'
                    resp = {'status': 'error', 'text': text, 'code': 2}
                elif has_video and not settings.FFMPEG_BIN:
                    text = 'Frames cannot be extracted (FFmpeg was not found)'
                    resp = {'status': 'error', 'text': text, 'code': 7}
            elif not (video.cache_file and os.path.isfile(video.cache_file)):
                resp = {'status': 'error',
                        'text': 'Initializing tracker...', 'code': 3}
            elif not video.project:
                resp = {'status': 'error', 'text': 'No project', 'code': 4}
            elif not video.project.labels.exists():
                resp = {'status': 'error',
                        'text': 'No labels in project', 'code': 5}
        return resp

    def get(self, request, id):
        resp = self.status(id)
        if resp['code'] in [2, 3]:
            video = Video.objects.get(id=id)
            for task_id in [video.extract_task_id, video.cache_task_id]:
                if task_id:
                    if is_task_running(task_id) or is_task_pending(task_id):
                        resp['status'] = 'wait'
                        break
                    elif task_failed(task_id):
                        resp['status'] = 'error'
                        resp['text'] = 'Something went wrong'
                        break
                    elif is_task_done(task_id):
                        resp['status'] = 'ok'
                    else:
                        resp['status'] = 'error'
                        resp['text'] = 'Task not scheduled. Is Celery running?'
                        break
        resp.pop('code', None)
        return JsonResponse(resp)

    def post(self, request, id):
        resp = self.status(id)
        video = Video.objects.get(id=id)
        if resp['code'] in [2, 3]:
            if resp['code'] == 2:
                if not (video.extract_task_id and is_task_running(video.extract_task_id)):
                    logger.debug('Creating extract chain')
                    task = chain(extract_frames.s() |
                                 create_cache_task.s())(id)
                    video.cache_task_id = task.task_id
                    video.extract_task_id = task.parent.task_id
                    video.save()
            elif resp['code'] == 3:
                if not (video.cache_task_id and is_task_running(video.cache_task_id)):
                    logger.debug('Creating cache task')
                    task = create_cache_task.delay(id)
                    video.cache_task_id = task.task_id
                    video.save()
            resp['status'] = 'wait'

        resp.pop('code', None)
        return JsonResponse(resp)


class ExportVideo(View):
    def get(self, request, vid):
        video = Video.objects.get(id=vid)
        if video.zipfile and os.path.exists(video.zipfile.path):
            with open(video.zipfile.path, 'rb') as f:
                response = HttpResponse(
                    f, content_type='application/x-zip-compressed')
                response['Content-Disposition'] = (
                    'attachment; '
                    'filename={}.zip'
                ).format(video.name)
        else:
            response = HttpResponseNotFound('File does not exist')
        return response


class ExportVideoStatus(View):
    def get(self, request):
        task_id = request.GET.get('task_id')
        if task_id:
            result = AsyncResult(task_id)
            if result.ready():
                resp = {'status': 'ok'}
            else:
                resp = {'status': 'wait', 'task_id': task_id}
        else:
            resp = {'status': 'error', 'text': 'task_id missing'}
        return JsonResponse(resp)

    def post(self, request):
        vid = request.POST['id']
        video = Video.objects.get(id=vid)
        resp = {}
        if video.zipfile and os.path.isfile(video.zipfile.path):
            resp['status'] = 'ok'
        elif video.image_list:
            task = create_zipfile.delay(vid)
            resp['status'] = 'wait'
            resp['task_id'] = task.task_id
            clean_zipfiles.delay()
        else:
            # TODO: if ffmpeg is installed, we could extract here
            resp = {
                'status': 'error',
                'text': 'Frames have not been extracted yet.'
            }
        return JsonResponse(resp)

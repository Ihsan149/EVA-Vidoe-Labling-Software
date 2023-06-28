import urllib
import urllib.request
import json
import uuid
import os
import cv2
import numpy as np
import h5py
import xml.etree.ElementTree as ET
import subprocess
import tempfile
import shutil
import logging
import zipfile
import linecache
from time import time

from django.core.files import File
from .models import Video, UploadFile, LabelMapping
from celery import shared_task
from django.conf import settings
from .utils import *
from .kcftracker import kcftracker

from celery.utils.log import get_task_logger
logger = get_task_logger(__name__)


class TrackerError(Exception):
    pass


class VideoError(Exception):
    pass


@shared_task
def tracker_task(video_id, frame_no, bbox):
    video = Video.objects.get(id=video_id)

    config = kcftracker.load_config(settings.TRACKER_SETTINGS)
    tracker = kcftracker.KCFTracker(config)

    hdf5_file = h5py.File(video.cache_file, 'r')

    global_scale = hdf5_file['scale'][0]
    bbox = [c*global_scale for c in bbox]  # The input image has been scaled
    coordinates = dict()
    coordinates[frame_no] = [c // global_scale for c in bbox]
    frame = hdf5_file['img'][frame_no, ...]
    tracker.init(bbox, frame)
    low = frame_no + 1
    last_frame = (frame_no//settings.TRACKER_SIZE + 1)*settings.TRACKER_SIZE
    num_images = hdf5_file['img'].shape[0]
    high = min(last_frame + 1, num_images)
    t0 = time()
    for i in range(low, high):
        frame = hdf5_file['img'][i, ...]
        ok, bbox = tracker.update(frame)
        bbox = list(bbox)
        if ok:
            # Tracking success
            coordinates[i] = [c//global_scale for c in bbox]
        else:
            # Tracking failure
            logger.error(
                "Something went wrong with the tracker, "
                "it was not able to update tracker with new frame"
            )
            # don't want to update a key frame if tracker fails, need to discuss
            # coordinates[frame_no + frame_counter] = bbox
            break
    t1 = time()
    logger.info('Time it took was: {}'.format(t1-t0))
    return coordinates


def scale_box(box, scale):
    (w, h) = box[2:]
    box = [a+b for (a, b) in zip(box, [(1-scale)/2*w, (1-scale)/2*h, 0, 0])]
    box = [a*b for (a, b) in zip(box, [1, 1, scale, scale])]
    return box


@shared_task
def create_cache_task(video_id):
    logger.info("went to create cache")
    video = Video.objects.get(id=video_id)

    if not (video.cache_file and os.path.isfile(video.cache_file)):
        files = video.uploadfile_set.filter(
            file_type=UploadFile.IMAGE).order_by('file')
        files = [os.path.join(settings.MEDIA_ROOT, x.file.name) for x in files]

        if len(files) == 0:
            raise TrackerError('No images. Cannot create cache.')

        img = cv2.imread(files[0])
        height = img.shape[0]
        width = img.shape[1]

        scale = min(((360 * 640) / (height * width)) ** 0.5, 1)

        new_size = (int(round(width * scale)), int(round(height * scale)))

        logger.info('Resizing ({}, {}) to {}'.format(width, height, new_size))

        path = os.path.join(settings.MEDIA_ROOT, 'cache')
        os.makedirs(path, exist_ok=True)
        cache_file_path = os.path.join(path, str(uuid.uuid4()) + '.hd5')

        hdf5_file = h5py.File(cache_file_path, 'w')
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]

        hdf5_file.create_dataset(
            'img', (len(files), height, width, channels), np.uint8)
        hdf5_file.create_dataset('scale', (1,), np.float32)

        hdf5_file['scale'][0] = scale
        file_amount = len(files)
        # Let's partition the work into batches to avoid memory issues
        batch_size = 100
        loop_till = round(file_amount / batch_size) if round(file_amount / batch_size) else 1
        for batch_index in range(0, loop_till):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, file_amount)
            images = list()
            if file_amount - end_index < batch_size / 2:
                end_index = file_amount
            for file in files[start_index:end_index]:
                img = cv2.imread(file)
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
                images.append(img)
            for k, img in enumerate(images):
                hdf5_file['img'][start_index + k, ...] = img[None]

        hdf5_file.close()
        video = Video.objects.get(id=video_id)
        video.cache_file = cache_file_path

    video.cache_task_id = ''
    video.save()

    return video_id


@shared_task
def extract_frames(video_id):
    img_ext = settings.IMAGE_FORMAT
    
    video = Video.objects.get(id=video_id)
    files = video.uploadfile_set.filter(file_type=UploadFile.VIDEO)

    if not files:
        raise VideoError('No video file, cannot extract frames')

    if video.uploadfile_set.filter(file_type=UploadFile.IMAGE).exists():
        raise VideoError('Images already exist')

    video_file = files[0].file.path
    
    img_dir = str(video.id)
    img_dir_abs = os.path.join(settings.MEDIA_ROOT, img_dir)
    
    try:
        subprocess.run([settings.FFMPEG_BIN, '-i', video_file,
                        os.path.join(img_dir_abs, '%04d.{}'.format(img_ext))], check=True)
    except subprocess.CalledProcessError:
        raise VideoError('Extraction frames from video failed')

    out_files = [os.path.join(img_dir, x)
                 for x in os.listdir(img_dir_abs) if x.endswith(img_ext)]
    for file in out_files:
        file_db = UploadFile.objects.create()
        with open(os.path.join(settings.MEDIA_ROOT, file), 'rb') as f:
            img_dimension = get_img_size_from_buffer(f)
        file_db.width = img_dimension[1]
        file_db.height = img_dimension[0]
        file_db.video = video
        file_db.file = file
        file_db.save()
        
    with open(os.path.join(settings.MEDIA_ROOT, out_files[0]), 'rb') as f:
        img_size = get_img_size_from_buffer(f)
        video.height = img_size[0]
        video.width = img_size[1]
        video.channels = img_size[2]
        video.save()

    video = Video.objects.get(id=video_id)
    video.extract_task_id = ''
    video.save()

    return video_id


def parse_annotations(annotation):
    labels = {}
    for track in json.loads(annotation):
        class_ = track['type']
        for frame in track['keyframes']:
            val = {x: frame[x] for x in ('x', 'y', 'w', 'h')}
            val.update({'class': class_})
            labels.setdefault(frame['frame'], []).append(val)
    return labels


def convert_to_darknet(video):
    result = {}
    if video.annotation:
        files = video.images
        
        width = video.width
        height = video.height
        
        assert(width != 0 and height != 0)
        
        label_mapping = LabelMapping.objects.filter(project=video.project)
        class_mapping = {x.label.name: x.num for x in label_mapping}
        
        labels = parse_annotations(video.annotation)
        
        for key, file in enumerate(files):
            if file[1] and file[2]:  # read width and height of each image
                width = file[1]
                height = file[2]
            filename = os.path.splitext(os.path.split(file[0])[1])[0] + '.txt'
            if key in labels:
                frame = []
                for label in labels[key]:
                    frame.append('{:d} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                        class_mapping[label['class']],
                        (label['x'] + label['w']/2) / width,
                        (label['y'] + label['h']/2) / height,
                        label['w'] / width,
                        label['h'] / height
                    ))
                text = '\n'.join(frame)
            else:
                text = ''
            result.update({filename: text})
    return result


def indent(elem, level=0):
    "See https://stackoverflow.com/questions/749796/pretty-printing-xml-in-python"
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def convert_to_pascal_voc(video):
    result = {}
    if video.annotation:
        files = video.images
        
        width = video.width
        height = video.height
        depth = video.channels
    
        labels = parse_annotations(video.annotation)

        for key, file in enumerate(files):
            if file[1] and file[2]:  # read width and height of each image
                width = file[1]
                height = file[2]
            filename = os.path.splitext(os.path.split(file[0])[1])[0] + '.xml'
            
            root = ET.Element('annotation')
            ET.SubElement(root, 'filename').text = os.path.split(file[0])[1]
            ET.SubElement(root, 'folder').text = video.name

            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(width)
            ET.SubElement(size, 'height').text = str(height)
            ET.SubElement(size, 'depth').text = str(depth)

            ET.SubElement(root, 'segmented').text = "0"
            
            if key in labels:
                for label in labels[key]:
                    obj = ET.SubElement(root, 'object')
                    name = ET.SubElement(obj, 'name')
                    name.text = label['class']

                    bbox = ET.SubElement(obj, 'bndbox')

                    bb = ET.SubElement(bbox, 'xmin')
                    bb.text = str(round(label['x']))
                    bb = ET.SubElement(bbox, 'ymin')
                    bb.text = str(round(label['y']))
                    bb = ET.SubElement(bbox, 'xmax')
                    bb.text = str(round(label['x'] + label['w']))
                    bb = ET.SubElement(bbox, 'ymax')
                    bb.text = str(round(label['y'] + label['h']))

                indent(root)
                text = ET.tostring(root, encoding='unicode')
            else:
                text = ''
            result.update({filename: text})
    return result


@shared_task
def create_zipfile(video_id):
    video = Video.objects.get(id=video_id)
    files = collect_images(video_id)

    os.makedirs(settings.ZIPFILE_ROOT, exist_ok=True)
    video.zipfile.name = os.path.join(
        settings.ZIPFILE_DIR, '.'.join([str(uuid.uuid4()), 'zip']))
    video.save()

    video_name = video.name
    with zipfile.ZipFile(video.zipfile.path, 'w', compression=zipfile.ZIP_STORED) as zip_file:
        for name, path in files.items():
            zip_file.write(path, os.path.join(video_name, name))

    return video_id


@shared_task
def clean_zipfiles():
    files = os.listdir(settings.ZIPFILE_ROOT)
    files = [os.path.join(settings.ZIPFILE_ROOT, x) for x in files if x.endswith('zip')]
    files.sort(key=lambda k: os.path.getmtime(k), reverse=True)
    files = files[settings.MAX_ZIPFILES:]
    for file in files:
        logger.debug('Deleting {}'.format(file))
        try:
            os.remove(file)
        except FileNotFoundError:
            pass


def collect_images(video_id):
    video = Video.objects.get(id=video_id)
    files = video.uploadfile_set.filter(file_type=UploadFile.IMAGE)
    files = {os.path.split(x.file.name)[1]: x.file.path for x in files}

    return files

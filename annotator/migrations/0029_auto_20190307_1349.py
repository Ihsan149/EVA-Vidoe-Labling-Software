# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2019-03-07 12:49
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('annotator', '0028_auto_20181106_1238'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='labellabelset',
            name='label',
        ),
        migrations.RemoveField(
            model_name='labellabelset',
            name='labelset',
        ),
        migrations.RemoveField(
            model_name='videolabels',
            name='label',
        ),
        migrations.RemoveField(
            model_name='videolabels',
            name='video',
        ),
        migrations.RemoveField(
            model_name='video',
            name='hidden_type',
        ),
        migrations.RemoveField(
            model_name='video',
            name='host',
        ),
        migrations.DeleteModel(
            name='LabelLabelset',
        ),
        migrations.DeleteModel(
            name='Videolabels',
        ),
    ]

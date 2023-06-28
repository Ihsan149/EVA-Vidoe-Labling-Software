# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-06-29 08:53
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('annotator', '0008_remove_labelset_color'),
    ]

    operations = [
        migrations.AddField(
            model_name='labelset',
            name='description',
            field=models.CharField(blank=True, default='description', help_text='description of the labelset for users convenience.', max_length=100, unique=True),
        ),
    ]
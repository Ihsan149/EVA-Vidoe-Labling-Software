@echo off

CALL venv\Scripts\activate.bat

start "django" python.exe manage.py runserver
start "celery" celery.exe -A eva worker --pool=solo -l info

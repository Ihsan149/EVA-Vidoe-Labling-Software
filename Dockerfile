FROM python:3.6-slim

RUN mkdir /app
WORKDIR /app

#apt-get install libglib2.0-dev
RUN apt-get update && apt-get install -y \
    libglib2.0-dev \
    libsm6 libxext6 libxrender-dev \
    ffmpeg \
    g++ \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir gunicorn psycopg2-binary

COPY . /app/
RUN mv /app/eva/deploy_settings /app/eva/deploy_settings.py
RUN python manage.py preparetracker

EXPOSE 8000

CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8000", "eva.wsgi"]

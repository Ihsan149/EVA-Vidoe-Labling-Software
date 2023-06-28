#!/bin/bash

set -e

BEAVERDAM_DIR=$(pwd)

# uwsgi needs write permission
chgrp -R www-data $BEAVERDAM_DIR
chmod -R g+w $BEAVERDAM_DIR
sudo find . -path ./venv -prune -o -type d -exec chmod g+s {} \;

apt-get -y update

apt-get -y install \
    nginx \
    python3 \
    python3-pip \
    python-opencv \
    build-essential

pip3 install --upgrade pip
pip3 install virtualenv
pip3 install uwsgi

#Install redis
cd /tmp
curl -O http://download.redis.io/redis-stable.tar.gz
tar xzvf redis-stable.tar.gz
cd redis-stable
make
make install

# Configure nginx
echo "Configure nginx"
rm -rf /etc/nginx/sites-enabled/default

sed -i "s,BEAVERDAM_DIR,$BEAVERDAM_DIR," $BEAVERDAM_DIR/deployment/nginx.conf
ln -sf $BEAVERDAM_DIR/deployment/nginx.conf /etc/nginx/sites-enabled/labeling_tool

# Configure uwsgi
echo "Configure uwsgi"
sed -i "s,BEAVERDAM_DIR,$BEAVERDAM_DIR," $BEAVERDAM_DIR/deployment/labeler_uwsgi.ini

mkdir -p /etc/uwsgi/vassals
cp $BEAVERDAM_DIR/deployment/emperor.ini /etc/uwsgi/
cp $BEAVERDAM_DIR/deployment/emperor.uwsgi.service /etc/systemd/system/
ln -sf $BEAVERDAM_DIR/deployment/labeler_uwsgi.ini /etc/uwsgi/vassals/labeler_uwsgi.ini

# Configure Celery
echo "Configure Celery"
mkdir -p /etc/conf.d

sed -i "s,BEAVERDAM_DIR,$BEAVERDAM_DIR," $BEAVERDAM_DIR/deployment/celery.conf
sed -i "s,BEAVERDAM_DIR,$BEAVERDAM_DIR," $BEAVERDAM_DIR/deployment/celery.service

cp $BEAVERDAM_DIR/deployment/celery.service /etc/systemd/system/
cp $BEAVERDAM_DIR/deployment/celery_tmp.conf /etc/tmpfiles.d/
ln -sf $BEAVERDAM_DIR/deployment/celery.conf /etc/conf.d/celery.conf


# Configure Redis
echo "Configure redis"
mkdir -p /etc/redis
cp $BEAVERDAM_DIR/deployment/redis.conf /etc/redis/
cp $BEAVERDAM_DIR/deployment/redis.service /etc/systemd/system/

adduser --system --group --no-create-home redis
mkdir -p /var/lib/redis
chown redis:redis /var/lib/redis
chmod 770 /var/lib/redis

# Start services

systemd-tmpfiles --create
systemctl enable nginx emperor.uwsgi celery redis
#systemctl start nginx emperor.uwsgi celery redis

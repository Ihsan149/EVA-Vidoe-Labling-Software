#!/bin/bash

trap 'killall' INT

killall() {
    trap '' INT TERM	# ignore INT and TERM while shutting down
    echo "**** Shutting down... ****"
    kill -TERM 0
    wait
    echo DONE
}

. venv/bin/activate

python manage.py runserver 0.0.0.0:8000 &
celery -A eva worker &
redis-server &

wait

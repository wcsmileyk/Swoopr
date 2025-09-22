#!/usr/bin/env bash
# Render build script

set -o errexit  # exit on error

pip install -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput
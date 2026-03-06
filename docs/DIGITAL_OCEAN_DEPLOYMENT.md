# Swoopr — Digital Ocean VPS Deployment Guide

## Overview

This guide covers migrating Swoopr from Render.com to a self-managed Digital Ocean Droplet.
The app is Django + PostGIS, served by Gunicorn behind Nginx, with a PostgreSQL database
running on the same droplet.

---

## Part 0: Code Change Required Before Deploying

### Remove the Render-specific settings block

In `Swoopr/settings.py`, the bottom of the file has a Render-specific override block:

```python
# Render.com production settings
if os.getenv('RENDER'):
    DEBUG = False
    ALLOWED_HOSTS = ['.onrender.com']
    ...
```

This needs to be removed. All those settings are already driven by environment variables
higher in the file — on the DO server you'll set them directly in the `.env` file.

**Delete lines 385–399** (the entire `if os.getenv('RENDER'):` block) from `settings.py`.

After removing it, commit and push the change before deploying.

---

## Part 1: Create the Droplet

1. Create a new Droplet in the Digital Ocean control panel:
   - **Image**: Ubuntu 24.04 LTS
   - **Size**: At minimum 2 GB RAM / 1 vCPU (4 GB recommended given numpy/pandas/scikit-learn)
   - **Region**: Choose one close to your users
   - **Authentication**: SSH key (strongly preferred over password)

2. Once the droplet is running, note its IP address.

3. Point your domain's DNS A record to that IP. (You can skip this initially and use the IP directly.)

---

## Part 2: Initial Server Setup

SSH in as root:

```bash
ssh root@YOUR_DROPLET_IP
```

### 2.1 Create a non-root user

```bash
adduser swoopr
usermod -aG sudo swoopr
# Copy your SSH key to the new user
rsync --archive --chown=swoopr:swoopr ~/.ssh /home/swoopr
```

Switch to the new user for all remaining steps:

```bash
su - swoopr
```

### 2.2 Update the system

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.3 Configure the firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

---

## Part 3: Install System Dependencies

### 3.1 Python

Ubuntu 24.04 ships with Python 3.12. Install pip and venv:

```bash
sudo apt install -y python3-pip python3-venv python3-dev
```

### 3.2 PostgreSQL and PostGIS

**This is the most critical dependency.** GeoDjango requires PostGIS on the database server.

```bash
sudo apt install -y postgresql postgresql-contrib postgis postgresql-16-postgis-3
```

> If your Ubuntu ships with a different PostgreSQL version, adjust `postgresql-16-postgis-3`
> accordingly (e.g. `postgresql-15-postgis-3`). Check with: `psql --version`

### 3.3 GDAL / GEOS / PROJ (required for GeoDjango)

```bash
sudo apt install -y gdal-bin libgdal-dev libgeos-dev libproj-dev
```

Verify GDAL is installed:

```bash
gdal-config --version
```

### 3.4 Nginx

```bash
sudo apt install -y nginx
```

### 3.5 Certbot (SSL — do this after DNS is pointed at the server)

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 3.6 Other build dependencies

```bash
sudo apt install -y build-essential libpq-dev git
```

---

## Part 4: Set Up the Database

### 4.1 Create the database and user

```bash
sudo -u postgres psql
```

Inside psql:

```sql
CREATE USER swoopr_user WITH PASSWORD 'your_strong_password_here';
CREATE DATABASE swoopr OWNER swoopr_user;
\c swoopr
CREATE EXTENSION postgis;
GRANT ALL PRIVILEGES ON DATABASE swoopr TO swoopr_user;
-- PostGIS tables need additional grants:
GRANT ALL ON TABLE spatial_ref_sys TO swoopr_user;
\q
```

### 4.2 Verify PostGIS

```bash
sudo -u postgres psql -d swoopr -c "SELECT PostGIS_version();"
```

---

## Part 5: Deploy the Application

### 5.1 Clone the repository

```bash
cd /var/www
sudo mkdir swoopr
sudo chown swoopr:swoopr swoopr
cd swoopr
git clone https://github.com/YOUR_ORG/YOUR_REPO.git .
```

Or copy files from your local machine:

```bash
# From your local machine:
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  /path/to/local/Swoopr/ swoopr@YOUR_DROPLET_IP:/var/www/swoopr/
```

### 5.2 Create the virtual environment

```bash
cd /var/www/swoopr
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.3 Create the environment file

```bash
cp .env.example .env
nano .env
```

Set these values (replace all placeholder values):

```dotenv
# Django
SECRET_KEY=generate-a-new-one-with-python-below
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Database
DB_NAME=swoopr
DB_USER=swoopr_user
DB_PASSWORD=your_strong_password_here
DB_HOST=localhost
DB_PORT=5432

# Security (enable these once SSL is working)
SECURE_SSL_REDIRECT=True
SECURE_HSTS_SECONDS=31536000
CSRF_COOKIE_SECURE=True
SESSION_COOKIE_SECURE=True

# ML
ML_ENABLED=False

# File uploads
FILE_UPLOAD_MAX_MEMORY_SIZE=5242880
DATA_UPLOAD_MAX_MEMORY_SIZE=5242880
```

Generate a SECRET_KEY:

```bash
source venv/bin/activate
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Paste the output as `SECRET_KEY` in your `.env`.

### 5.4 Create required directories

```bash
mkdir -p /var/www/swoopr/logs
mkdir -p /var/www/swoopr/media
```
python
### 5.5 Run the build steps

```bash
source venv/bin/activate
python manage.py migrate
python manage.py collectstatic --noinput
```

### 5.6 Create a Django superuser (optional)

```bash
python manage.py createsuperuser
```

---

## Part 6: Configure Gunicorn as a Systemd Service

### 6.1 Create the socket file

```bash
sudo nano /etc/systemd/system/swoopr.socket
```

```ini
[Unit]
Description=Swoopr Gunicorn Socket

[Socket]
ListenStream=/run/swoopr.sock

[Install]
WantedBy=sockets.target
```

### 6.2 Create the service file

```bash
sudo nano /etc/systemd/system/swoopr.service
```

```ini
[Unit]
Description=Swoopr Gunicorn Daemon
Requires=swoopr.socket
After=network.target

[Service]
User=swoopr
Group=www-data
WorkingDirectory=/var/www/swoopr
EnvironmentFile=/var/www/swoopr/.env
ExecStart=/var/www/swoopr/venv/bin/gunicorn \
          --access-logfile - \
          --workers 3 \
          --bind unix:/run/swoopr.sock \
          Swoopr.wsgi:application

[Install]
WantedBy=multi-user.target
```

> **Workers**: A good rule of thumb is `(2 × CPU cores) + 1`. For a 1-vCPU droplet, 3 workers
> is appropriate. Increase if you have more cores.

### 6.3 Enable and start the service

```bash
sudo systemctl daemon-reload
sudo systemctl start swoopr.socket
sudo systemctl enable swoopr.socket
sudo systemctl start swoopr
sudo systemctl enable swoopr
```

Check it is running:

```bash
sudo systemctl status swoopr
```

---

## Part 7: Configure Nginx

### 7.1 Create the site config

```bash
sudo nano /etc/nginx/sites-available/swoopr
```

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    client_max_body_size 10M;

    location = /favicon.ico { access_log off; log_not_found off; }

    location /static/ {
        root /var/www/swoopr/staticfiles;
    }

    location /media/ {
        root /var/www/swoopr;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:/run/swoopr.sock;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

> Note: WhiteNoise is already configured in the app to serve static files, so this Nginx
> `location /static/` block is optional but recommended for better performance.

### 7.2 Enable the site

```bash
sudo ln -s /etc/nginx/sites-available/swoopr /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Part 8: SSL with Let's Encrypt

Once your DNS A record is pointed at the droplet and has propagated:

```bash
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

Certbot will automatically modify your Nginx config to handle HTTPS and set up auto-renewal.

Verify auto-renewal works:

```bash
sudo certbot renew --dry-run
```

---

## Part 9: Ongoing Deployment (Updating the App)

When you push new code, SSH into the server and run:

```bash
cd /var/www/swoopr
git pull origin master
source venv/bin/activate
pip install -r requirements.txt   # only if requirements changed
python manage.py migrate           # only if there are new migrations
python manage.py collectstatic --noinput
sudo systemctl restart swoopr
```

---

## Part 10: Useful Commands

| Purpose | Command |
|---|---|
| View app logs | `sudo journalctl -u swoopr -f` |
| View Nginx logs | `sudo tail -f /var/log/nginx/error.log` |
| View app error logs | `tail -f /var/www/swoopr/logs/errors.log` |
| Restart app | `sudo systemctl restart swoopr` |
| Reload Nginx | `sudo systemctl reload nginx` |
| Check Gunicorn socket | `sudo systemctl status swoopr.socket` |
| Connect to DB | `sudo -u postgres psql -d swoopr` |

---

## Differences from Render

| Feature | Render | Digital Ocean VPS |
|---|---|---|
| Process management | Managed by Render | systemd |
| Port binding | Dynamic `$PORT` env var | Unix socket → Nginx |
| SSL | Automatic | Certbot (manual setup, auto-renewal) |
| Static files | WhiteNoise | WhiteNoise (or Nginx direct) |
| Database | Managed Render DB | Self-managed PostgreSQL on same droplet |
| Deploys | Auto from git push | Manual `git pull` + restart |
| Env vars | Set in Render dashboard | `.env` file on server |
| Production flag | `RENDER=true` env var | Set vars directly in `.env` |

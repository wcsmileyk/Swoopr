#!/usr/bin/env python3
"""
Management command to check and collect static files properly
"""

from django.core.management.base import BaseCommand
from django.core.management import call_command
from django.conf import settings
from django.contrib.staticfiles import finders
import os


class Command(BaseCommand):
    help = 'Check static files configuration and collect files'

    def add_arguments(self, parser):
        parser.add_argument(
            '--collect',
            action='store_true',
            help='Collect static files after checking configuration',
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing static files before collecting',
        )
        parser.add_argument(
            '--check-admin',
            action='store_true',
            help='Check specifically for Django admin static files',
        )

    def handle(self, *args, **options):
        collect = options['collect']
        clear = options['clear']
        check_admin = options['check_admin']

        self.stdout.write("🔍 Checking static files configuration...")
        self.stdout.write("=" * 50)

        # Check basic configuration
        self.stdout.write(f"📁 STATIC_URL: {settings.STATIC_URL}")
        self.stdout.write(f"📁 STATIC_ROOT: {settings.STATIC_ROOT}")
        self.stdout.write(f"📁 DEBUG: {settings.DEBUG}")

        # Check storage backend
        storages = getattr(settings, 'STORAGES', {})
        static_backend = storages.get('staticfiles', {}).get('BACKEND', 'Not configured')
        self.stdout.write(f"🔧 Static storage backend: {static_backend}")

        # Check STATICFILES_DIRS
        static_dirs = getattr(settings, 'STATICFILES_DIRS', [])
        self.stdout.write(f"📂 STATICFILES_DIRS: {static_dirs}")

        # Check if STATIC_ROOT exists
        if settings.STATIC_ROOT.exists():
            file_count = len(list(settings.STATIC_ROOT.rglob('*')))
            self.stdout.write(f"✅ STATIC_ROOT exists with {file_count} files")
        else:
            self.stdout.write(f"⚠️  STATIC_ROOT does not exist")

        # Check admin static files specifically
        if check_admin:
            self.stdout.write(f"\n🔍 Checking Django admin static files...")
            admin_css = finders.find('admin/css/base.css')
            if admin_css:
                self.stdout.write(f"✅ Found admin CSS: {admin_css}")
            else:
                self.stdout.write(f"❌ Admin CSS not found!")

            admin_js = finders.find('admin/js/core.js')
            if admin_js:
                self.stdout.write(f"✅ Found admin JS: {admin_js}")
            else:
                self.stdout.write(f"❌ Admin JS not found!")

        # Check WhiteNoise settings
        self.stdout.write(f"\n🔧 WhiteNoise Configuration:")
        self.stdout.write(f"   USE_FINDERS: {getattr(settings, 'WHITENOISE_USE_FINDERS', False)}")
        self.stdout.write(f"   MANIFEST_STRICT: {getattr(settings, 'WHITENOISE_MANIFEST_STRICT', True)}")
        self.stdout.write(f"   AUTOREFRESH: {getattr(settings, 'WHITENOISE_AUTOREFRESH', False)}")

        # Collect static files if requested
        if collect:
            self.stdout.write(f"\n📦 Collecting static files...")

            if clear:
                self.stdout.write(f"🗑️  Clearing existing static files...")
                call_command('collectstatic', '--clear', '--noinput', verbosity=2)
            else:
                call_command('collectstatic', '--noinput', verbosity=2)

            # Verify admin files were collected
            admin_css_path = settings.STATIC_ROOT / 'admin' / 'css' / 'base.css'
            if admin_css_path.exists():
                self.stdout.write(f"✅ Admin CSS collected successfully")
            else:
                self.stdout.write(f"❌ Admin CSS was not collected!")

        self.stdout.write(f"\n💡 Recommendations:")
        if not settings.DEBUG:
            self.stdout.write(f"   🚀 Production mode detected")
            self.stdout.write(f"   📦 Run: python manage.py check_static --collect")
            self.stdout.write(f"   🔄 Restart your web server after collecting")
        else:
            self.stdout.write(f"   🛠️  Development mode detected")
            self.stdout.write(f"   📦 Static files served directly by Django")

        if static_backend == 'whitenoise.storage.CompressedManifestStaticFilesStorage':
            self.stdout.write(f"   ⚠️  Using strict manifest storage")
            self.stdout.write(f"   💡 Consider using CompressedStaticFilesStorage for production")

        self.stdout.write(f"\n✅ Static files check complete!")
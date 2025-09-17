import os
import glob
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from django.db import transaction
from flights.flight_manager import process_flysight_file
from flights.models import Flight


class Command(BaseCommand):
    help = 'Import FlySight CSV files from a directory tree'

    def add_arguments(self, parser):
        parser.add_argument(
            'directory',
            type=str,
            help='Root directory to search for CSV files'
        )
        parser.add_argument(
            '--username',
            type=str,
            default='smiley',
            help='Username to associate flights with (default: smiley)'
        )
        parser.add_argument(
            '--skip-existing',
            action='store_true',
            help='Skip files that have already been processed'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be processed without actually importing'
        )

    def handle(self, *args, **options):
        directory = options['directory']
        username = options['username']
        skip_existing = options['skip_existing']
        dry_run = options['dry_run']

        # Validate directory exists
        if not os.path.exists(directory):
            raise CommandError(f'Directory does not exist: {directory}')

        # Get user
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            raise CommandError(f'User does not exist: {username}')

        self.stdout.write(f'Searching for CSV files in: {directory}')
        self.stdout.write(f'Importing for user: {username}')

        # Find all CSV files recursively
        csv_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, file))

        self.stdout.write(f'Found {len(csv_files)} CSV files')

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - No files will be processed'))
            for csv_file in csv_files:
                self.stdout.write(f'Would process: {csv_file}')
            return

        # Track statistics
        processed = 0
        skipped = 0
        errors = 0
        swoops_found = 0

        for csv_file in csv_files:
            try:
                # Check if file already processed
                if skip_existing:
                    # Extract potential session ID from filename or path
                    filename = os.path.basename(csv_file)
                    if Flight.objects.filter(pilot=user, session_id__icontains=filename[:8]).exists():
                        self.stdout.write(f'Skipping (already exists): {csv_file}')
                        skipped += 1
                        continue

                self.stdout.write(f'Processing: {csv_file}')

                # Process the file
                flight = process_flysight_file(csv_file, pilot=user)

                if flight.is_swoop:
                    swoops_found += 1
                    status = f'SWOOP ({flight.turn_rotation:.0f}°)'
                else:
                    status = 'FLIGHT'

                if flight.analysis_successful:
                    self.stdout.write(
                        self.style.SUCCESS(f'✓ {status} - {flight.session_id[:8]}')
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(f'⚠ {status} - Analysis failed: {flight.analysis_error}')
                    )

                processed += 1

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'✗ Failed to process {csv_file}: {str(e)}')
                )
                errors += 1
                continue

        # Print summary
        self.stdout.write('\n' + '='*50)
        self.stdout.write(self.style.SUCCESS('IMPORT SUMMARY'))
        self.stdout.write('='*50)
        self.stdout.write(f'Total files found: {len(csv_files)}')
        self.stdout.write(f'Successfully processed: {processed}')
        self.stdout.write(f'Skipped: {skipped}')
        self.stdout.write(f'Errors: {errors}')
        self.stdout.write(f'Swoops detected: {swoops_found}')

        if processed > 0:
            success_rate = (processed / (processed + errors)) * 100
            self.stdout.write(f'Success rate: {success_rate:.1f}%')

        self.stdout.write('='*50)
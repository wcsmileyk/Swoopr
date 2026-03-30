"""
Management command to fix max_vertical_speed and max_ground_speed on all swoop flights.

Previous algorithm stored 3-second rolling window averages. This command replaces
those with the true instantaneous values at the already-identified peak indices
(max_vspeed_idx / max_gspeed_idx), which are already correct.

Much faster than a full reanalyze — no detection logic runs, just an index lookup.
"""

from django.core.management.base import BaseCommand
from django.db import transaction
from flights.models import Flight

MPH_PER_MPS = 2.23694


class Command(BaseCommand):
    help = 'Fix peak speeds: replace 3-second window averages with true instantaneous values'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would change without saving',
        )
        parser.add_argument(
            '--pilot',
            type=str,
            default=None,
            help='Limit to a specific pilot username',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        pilot = options['pilot']

        qs = Flight.objects.filter(
            is_swoop=True,
            swoop_rejected=False,
            analysis_successful=True,
            max_vspeed_idx__isnull=False,
            max_gspeed_idx__isnull=False,
        )
        if pilot:
            qs = qs.filter(pilot__username=pilot)
            self.stdout.write(f'Filtering to pilot: {pilot}')

        total = qs.count()
        self.stdout.write(f'Flights to process: {total}')
        if dry_run:
            self.stdout.write('DRY RUN — no changes will be saved\n')

        updated = 0
        skipped = 0
        errors = 0

        for flight in qs.iterator(chunk_size=100):
            gps_data = flight.get_gps_data()
            if not gps_data:
                self.stdout.write(
                    self.style.WARNING(f'  Flight {flight.id}: no GPS data, skipping')
                )
                skipped += 1
                continue

            vspeed_idx = flight.max_vspeed_idx
            gspeed_idx = flight.max_gspeed_idx

            if vspeed_idx >= len(gps_data) or gspeed_idx >= len(gps_data):
                self.stdout.write(
                    self.style.WARNING(
                        f'  Flight {flight.id}: index out of range '
                        f'(vspeed={vspeed_idx}, gspeed={gspeed_idx}, points={len(gps_data)}), skipping'
                    )
                )
                skipped += 1
                continue

            try:
                new_vspeed_ms = abs(float(gps_data[vspeed_idx]['velocity_down']))
                new_gspeed_ms = float(gps_data[gspeed_idx]['ground_speed'])
            except (KeyError, TypeError, ValueError) as e:
                self.stdout.write(
                    self.style.ERROR(f'  Flight {flight.id}: bad GPS data — {e}')
                )
                errors += 1
                continue

            old_vspeed = flight.max_vertical_speed_mph or 0.0
            old_gspeed = flight.max_ground_speed_mph or 0.0
            new_vspeed_mph = new_vspeed_ms * MPH_PER_MPS
            new_gspeed_mph = new_gspeed_ms * MPH_PER_MPS

            delta_v = new_vspeed_mph - old_vspeed
            delta_g = new_gspeed_mph - old_gspeed

            self.stdout.write(
                f'  Flight {flight.id} (pilot={flight.pilot.username}): '
                f'vspeed {old_vspeed:.1f}→{new_vspeed_mph:.1f} mph ({delta_v:+.1f})  '
                f'gspeed {old_gspeed:.1f}→{new_gspeed_mph:.1f} mph ({delta_g:+.1f})'
            )

            if not dry_run:
                with transaction.atomic():
                    flight.max_vertical_speed_ms = new_vspeed_ms
                    flight.max_ground_speed_ms = new_gspeed_ms
                    flight.max_vertical_speed_mph = new_vspeed_mph
                    flight.max_ground_speed_mph = new_gspeed_mph
                    flight.save(update_fields=[
                        'max_vertical_speed_ms',
                        'max_ground_speed_ms',
                        'max_vertical_speed_mph',
                        'max_ground_speed_mph',
                    ])

            updated += 1

        self.stdout.write('')
        self.stdout.write(f'Updated : {updated}')
        self.stdout.write(f'Skipped : {skipped}')
        self.stdout.write(f'Errors  : {errors}')
        if dry_run:
            self.stdout.write('(dry run — nothing saved)')

"""
Backfills canopy_start_timestamp on existing GPS flights and recomputes
straight-line canopy performance stats for all canopies.

Usage:
    python manage.py backfill_canopy_stats
    python manage.py backfill_canopy_stats --flights-only
    python manage.py backfill_canopy_stats --stats-only
    python manage.py backfill_canopy_stats --force   # re-detect even if timestamp already set
"""

import pandas as pd
from django.core.management.base import BaseCommand

from flights.flight_manager import FlightManager
from flights.models import Flight
from users.canopy_stats import update_canopy_stats
from users.models import Canopy


class Command(BaseCommand):
    help = 'Backfill canopy_start_timestamp on flights and recompute canopy stats'

    def add_arguments(self, parser):
        parser.add_argument(
            '--flights-only',
            action='store_true',
            help='Only backfill canopy_start_timestamp; skip canopy stats update',
        )
        parser.add_argument(
            '--stats-only',
            action='store_true',
            help='Only recompute canopy stats; skip timestamp backfill',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Re-detect canopy start even for flights that already have a timestamp',
        )

    def handle(self, *args, **options):
        flights_only = options['flights_only']
        stats_only = options['stats_only']
        force = options['force']

        if not stats_only:
            self._backfill_timestamps(force)

        if not flights_only:
            self._recompute_stats()

    def _backfill_timestamps(self, force):
        manager = FlightManager()

        qs = Flight.objects.filter(
            analysis_successful=True,
            gps_data_compressed__isnull=False,
        )
        if not force:
            qs = qs.filter(canopy_start_timestamp__isnull=True)

        total = qs.count()
        self.stdout.write(f'Backfilling canopy start timestamp on {total} flights...')

        ok = 0
        failed = 0
        for flight in qs.iterator():
            try:
                gps_data = flight.get_gps_data()
                if not gps_data or len(gps_data) < 20:
                    continue

                df = pd.DataFrame(gps_data)
                df['t_s'] = df['timestamp'] - df['timestamp'].iloc[0]
                df['AGL'] = pd.to_numeric(df['altitude_agl'], errors='coerce')
                df['velD'] = pd.to_numeric(df['velocity_down'], errors='coerce')
                if 'velocity_north' in df.columns:
                    df['velN'] = pd.to_numeric(df['velocity_north'], errors='coerce')
                if 'velocity_east' in df.columns:
                    df['velE'] = pd.to_numeric(df['velocity_east'], errors='coerce')
                df = df.dropna(subset=['t_s', 'AGL', 'velD']).reset_index(drop=True)

                if len(df) < 20:
                    continue

                landing_idx = len(df) - 1
                canopy_idx = manager.find_canopy_start_idx(df, landing_idx)
                flight.canopy_start_timestamp = float(df.iloc[canopy_idx]['timestamp'])
                flight.save(update_fields=['canopy_start_timestamp'])
                ok += 1
                self.stdout.write(f'  Flight {flight.id}: canopy start at index {canopy_idx}')
            except Exception as e:
                failed += 1
                self.stdout.write(
                    self.style.WARNING(f'  Flight {flight.id} FAILED: {e}')
                )

        self.stdout.write(self.style.SUCCESS(
            f'Timestamp backfill complete: {ok} updated, {failed} failed.'
        ))

    def _recompute_stats(self):
        canopies = Canopy.objects.all()
        total = canopies.count()
        self.stdout.write(f'\nRecomputing canopy stats for {total} canopies...')

        for canopy in canopies:
            try:
                n = update_canopy_stats(canopy)
                self.stdout.write(f'  {canopy}: {n} flight(s) contributed')
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f'  {canopy} FAILED: {e}')
                )

        self.stdout.write(self.style.SUCCESS('Canopy stats recompute complete.'))

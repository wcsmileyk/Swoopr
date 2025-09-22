#!/usr/bin/env python3
"""
Simplified re-analysis command that manually performs the improved turn detection
without going through the full analyze_swoop pipeline.
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone
from flights.models import Flight
from flights.flight_manager import FlightManager
import pandas as pd
import numpy as np


class Command(BaseCommand):
    help = 'Simple re-analysis of flights using improved turn detection'

    def add_arguments(self, parser):
        parser.add_argument(
            '--flight-id',
            type=int,
            required=True,
            help='Re-analyze a specific flight by ID',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show analysis results without saving',
        )

    def handle(self, *args, **options):
        flight_id = options['flight_id']
        dry_run = options['dry_run']

        try:
            flight = Flight.objects.get(id=flight_id)
        except Flight.DoesNotExist:
            raise CommandError(f'Flight with ID {flight_id} does not exist')

        self.stdout.write(f'Analyzing flight {flight_id} (Pilot: {flight.pilot.username})')

        # Get GPS data
        gps_data = flight.get_gps_data()
        if not gps_data:
            self.stdout.write(
                self.style.ERROR('No GPS data available for this flight')
            )
            return

        # Reconstruct DataFrame
        df_data = {
            't_s': [i * 0.2 for i in range(len(gps_data))],
            'lat': [point.get('lat', 0) for point in gps_data],
            'lon': [point.get('lon', 0) for point in gps_data],
            'hMSL': [point.get('altitude_msl', 0) for point in gps_data],
            'velN': [point.get('velocity_north', 0) for point in gps_data],
            'velE': [point.get('velocity_east', 0) for point in gps_data],
            'velD': [point.get('velocity_down', 0) for point in gps_data],
            'hAcc': [point.get('h_acc', 0) for point in gps_data],
            'vAcc': [point.get('v_acc', 0) for point in gps_data],
            'sAcc': [point.get('s_acc', 0) for point in gps_data],
            'numSV': [point.get('num_sv', 0) for point in gps_data],
            'AGL': [point.get('altitude_agl', 0) for point in gps_data],
            'gspeed': [point.get('ground_speed', 0) for point in gps_data],
        }

        df = pd.DataFrame(df_data)

        # Calculate heading
        from flights.flight_manager import compute_heading
        df['heading'] = compute_heading(df)

        # Initialize flight manager
        manager = FlightManager()

        try:
            # Manually perform the key analysis steps
            landing_idx = manager.get_landing(df)
            self.stdout.write(f'Landing detected at index: {landing_idx}')

            # Try traditional flare detection first
            try:
                flare_idx = manager.find_flare(df, landing_idx)
                flare_method = "traditional"
                self.stdout.write(f'Traditional flare detected at index: {flare_idx}')
            except (ValueError, IndexError) as e:
                self.stdout.write(f'Traditional flare detection failed: {e}')
                # Use fallback method
                flare_idx = manager.find_turn_start_fallback(df, landing_idx)
                flare_method = "turn_detection"
                self.stdout.write(f'Fallback turn detection at index: {flare_idx}')

            # Find max speeds
            max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)
            self.stdout.write(f'Max vertical speed at index: {max_vspeed_idx}')
            self.stdout.write(f'Max ground speed at index: {max_gspeed_idx}')

            # Calculate rotation
            turn_rotation = manager.get_rotation(df, flare_idx, max_gspeed_idx)
            self.stdout.write(f'Turn rotation: {turn_rotation:.1f}°')

            # Get rollout
            rollout_start_idx, rollout_end_idx = manager.get_roll_out(df, max_vspeed_idx, max_gspeed_idx, landing_idx)

            # Calculate metrics
            max_vspeed_ms = abs(df.iloc[max_vspeed_idx]['velD'])
            max_gspeed_ms = df.iloc[max_gspeed_idx]['gspeed']

            self.stdout.write('\n' + '='*50)
            self.stdout.write('ANALYSIS RESULTS')
            self.stdout.write('='*50)
            self.stdout.write(f'Flare method: {flare_method}')
            self.stdout.write(f'Turn rotation: {turn_rotation:.1f}° {"left" if turn_rotation < 0 else "right"}')
            self.stdout.write(f'Max vertical speed: {max_vspeed_ms * 2.23694:.1f} mph')
            self.stdout.write(f'Max ground speed: {max_gspeed_ms * 2.23694:.1f} mph')

            if dry_run:
                self.stdout.write(
                    self.style.WARNING('\nDRY RUN - No changes saved to database')
                )
            else:
                # Update flight record manually
                with transaction.atomic():
                    flight.is_swoop = True
                    flight.landing_detected = True
                    flight.analysis_successful = True
                    flight.analysis_error = ''

                    # Store indices
                    flight.landing_idx = landing_idx
                    flight.flare_idx = flare_idx
                    flight.max_vspeed_idx = max_vspeed_idx
                    flight.max_gspeed_idx = max_gspeed_idx
                    flight.rollout_start_idx = rollout_start_idx
                    flight.rollout_end_idx = rollout_end_idx

                    # Store detection method
                    flight.flare_detection_method = flare_method

                    # Store metrics
                    flight.turn_rotation = turn_rotation
                    flight.turn_direction = "left" if turn_rotation < 0 else "right"
                    flight.max_vertical_speed_ms = max_vspeed_ms
                    flight.max_ground_speed_ms = max_gspeed_ms
                    flight.max_vertical_speed_mph = max_vspeed_ms * 2.23694
                    flight.max_ground_speed_mph = max_gspeed_ms * 2.23694

                    # Store altitudes
                    flight.flare_altitude_agl = df.iloc[flare_idx]['AGL']
                    flight.max_vspeed_altitude_agl = df.iloc[max_vspeed_idx]['AGL']
                    flight.max_gspeed_altitude_agl = df.iloc[max_gspeed_idx]['AGL']
                    flight.landing_altitude_agl = df.iloc[landing_idx]['AGL']

                    # Set analysis timestamp
                    flight.analyzed_at = timezone.now()

                    # Save with specific fields to avoid conflicts
                    flight.save()

                self.stdout.write(
                    self.style.SUCCESS('\nSUCCESS - Flight analysis updated!')
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Analysis failed: {str(e)}')
            )
            raise e
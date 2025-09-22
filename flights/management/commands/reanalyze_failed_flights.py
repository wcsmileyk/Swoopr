#!/usr/bin/env python3
"""
Management command to re-analyze flights that previously failed analysis.
This is particularly useful after improving the turn detection algorithm
to catch flights that didn't have traditional flares.
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone
from flights.models import Flight
from flights.flight_manager import FlightManager
import tempfile
import os


class Command(BaseCommand):
    help = 'Re-analyze flights that previously failed analysis using the improved turn detection algorithm'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be re-analyzed without actually doing it',
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Limit the number of flights to re-analyze (useful for testing)',
        )
        parser.add_argument(
            '--flight-id',
            type=int,
            help='Re-analyze a specific flight by ID',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Re-analyze all flights, including those that were previously successful',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']
        flight_id = options['flight_id']
        force = options['force']

        # Build the queryset
        if flight_id:
            flights = Flight.objects.filter(id=flight_id)
            if not flights.exists():
                raise CommandError(f'Flight with ID {flight_id} does not exist')
        elif force:
            flights = Flight.objects.all()
            self.stdout.write(
                self.style.WARNING('FORCE mode: Re-analyzing ALL flights')
            )
        else:
            flights = Flight.objects.filter(analysis_successful=False)

        if limit:
            flights = flights[:limit]

        total_count = flights.count()

        if total_count == 0:
            self.stdout.write(
                self.style.SUCCESS('No flights need re-analysis!')
            )
            return

        self.stdout.write(
            f'Found {total_count} flight(s) to re-analyze'
        )

        if dry_run:
            self.stdout.write(
                self.style.WARNING('DRY RUN MODE - No changes will be made')
            )
            for flight in flights:
                self.stdout.write(
                    f'Would re-analyze: Flight {flight.id} (Pilot: {flight.pilot.username}, '
                    f'Error: {flight.analysis_error[:50] if flight.analysis_error else "No error"}...)'
                )
            return

        # Confirm before proceeding
        if not force and total_count > 10:
            confirm = input(f'Are you sure you want to re-analyze {total_count} flights? (y/N): ')
            if confirm.lower() != 'y':
                self.stdout.write('Aborted.')
                return

        # Initialize the flight manager
        manager = FlightManager()

        success_count = 0
        error_count = 0
        skipped_count = 0

        for i, flight in enumerate(flights, 1):
            self.stdout.write(
                f'[{i}/{total_count}] Re-analyzing flight {flight.id} '
                f'(Pilot: {flight.pilot.username})...',
                ending=''
            )

            try:
                # Check if flight has GPS data
                gps_data = flight.get_gps_data()
                if not gps_data:
                    self.stdout.write(
                        self.style.WARNING(' SKIPPED - No GPS data available')
                    )
                    skipped_count += 1
                    continue

                # Convert GPS data back to DataFrame format for analysis
                import pandas as pd
                import numpy as np

                # Reconstruct the DataFrame from stored GPS data to match original FlySight format
                # Use relative timestamps to avoid datetime conflicts
                import datetime

                # Use a simple sequential timestamp to avoid datetime parsing issues
                base_timestamp = 1000000000.0  # Simple base timestamp

                df_data = {
                    '$GNSS': ['$GPRMC'] * len(gps_data),  # Standard GNSS type
                    'time': [base_timestamp + i * 0.2 for i in range(len(gps_data))],  # Sequential timestamps
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
                    # Calculated columns that are added by the flight manager
                    't_s': [i * 0.2 for i in range(len(gps_data))],  # Relative time in seconds
                    'AGL': [point.get('altitude_agl', 0) for point in gps_data],
                    'gspeed': [point.get('ground_speed', 0) for point in gps_data],
                }

                df = pd.DataFrame(df_data)

                # Calculate derived columns that the flight manager expects
                from flights.flight_manager import compute_heading
                df['heading'] = compute_heading(df)

                # Store original values for comparison
                original_is_swoop = flight.is_swoop
                original_analysis_successful = flight.analysis_successful
                original_error = flight.analysis_error

                # Re-run the analysis (the analyze_swoop method includes flight.save())
                with transaction.atomic():
                    try:
                        # Temporarily remove any problematic datetime fields
                        original_analyzed_at = flight.analyzed_at
                        flight.analyzed_at = None

                        # Run the analysis
                        manager.analyze_swoop(flight, df)

                    except Exception as analysis_error:
                        # If analysis fails, restore original state and re-raise
                        flight.analyzed_at = original_analyzed_at
                        raise analysis_error

                # Report results
                if flight.analysis_successful:
                    if flight.is_swoop:
                        detection_method = getattr(flight, 'flare_detection_method', 'unknown')
                        self.stdout.write(
                            self.style.SUCCESS(
                                f' SUCCESS - Swoop detected! '
                                f'({flight.turn_rotation:.0f}° {flight.turn_direction}, '
                                f'method: {detection_method})'
                            )
                        )
                    else:
                        self.stdout.write(
                            self.style.SUCCESS(' SUCCESS - Non-swoop flight')
                        )
                    success_count += 1
                else:
                    self.stdout.write(
                        self.style.ERROR(f' FAILED - {flight.analysis_error}')
                    )
                    error_count += 1

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f' ERROR - {str(e)}')
                )
                error_count += 1
                # Restore original state on error
                flight.is_swoop = original_is_swoop
                flight.analysis_successful = original_analysis_successful
                flight.analysis_error = original_error
                flight.save()

        # Summary
        self.stdout.write('\n' + '='*50)
        self.stdout.write('RE-ANALYSIS SUMMARY')
        self.stdout.write('='*50)
        self.stdout.write(f'Total flights processed: {total_count}')
        self.stdout.write(
            self.style.SUCCESS(f'Successful re-analyses: {success_count}')
        )
        if error_count > 0:
            self.stdout.write(
                self.style.ERROR(f'Failed re-analyses: {error_count}')
            )
        if skipped_count > 0:
            self.stdout.write(
                self.style.WARNING(f'Skipped (no data): {skipped_count}')
            )

        # Show improvement in success rate if we processed failed flights
        if not force and not flight_id:
            new_success_rate = Flight.objects.filter(analysis_successful=True).count() / Flight.objects.count() * 100
            self.stdout.write(
                f'New overall success rate: {new_success_rate:.1f}%'
            )
#!/usr/bin/env python3
"""
Management command to bulk re-analyze all flights with the latest algorithms.
This is particularly useful after major algorithm improvements (like ML integration)
to update the entire database with better rotation predictions.
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone
from flights.models import Flight
from flights.flight_manager import FlightManager
import pandas as pd
import numpy as np


class Command(BaseCommand):
    help = 'Bulk re-analyze all flights with the latest algorithms (including ML)'

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
            '--pilot',
            type=str,
            help='Re-analyze flights for a specific pilot username',
        )
        parser.add_argument(
            '--failed-only',
            action='store_true',
            help='Only re-analyze previously failed flights',
        )
        parser.add_argument(
            '--swoops-only',
            action='store_true',
            help='Only re-analyze flights already marked as swoops',
        )
        parser.add_argument(
            '--force-overwrite',
            action='store_true',
            help='Overwrite even successful analyses (use for algorithm updates)',
        )
        parser.add_argument(
            '--skip-ml',
            action='store_true',
            help='Skip ML predictions (faster, traditional algorithm only)',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Process flights in batches of this size',
        )
        parser.add_argument(
            '--show-detailed-ml',
            action='store_true',
            help='Show detailed ML vs traditional comparisons for all metrics',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']
        pilot = options['pilot']
        failed_only = options['failed_only']
        swoops_only = options['swoops_only']
        force_overwrite = options['force_overwrite']
        skip_ml = options['skip_ml']
        batch_size = options['batch_size']
        show_detailed_ml = options['show_detailed_ml']

        # Build the queryset
        flights = Flight.objects.all()

        if pilot:
            flights = flights.filter(pilot__username=pilot)
            self.stdout.write(f'Filtering by pilot: {pilot}')

        if failed_only:
            flights = flights.filter(analysis_successful=False)
            self.stdout.write('Filtering to failed flights only')

        if swoops_only:
            flights = flights.filter(is_swoop=True)
            self.stdout.write('Filtering to swoops only')

        if limit:
            flights = flights[:limit]

        total_count = flights.count()

        if total_count == 0:
            self.stdout.write(
                self.style.SUCCESS('No flights match the criteria!')
            )
            return

        # Initialize the flight manager
        manager = FlightManager()

        # Check ML model status
        ml_status = "✅ LOADED" if manager.ml_model_loaded else "❌ NOT AVAILABLE"
        if skip_ml:
            ml_status += " (SKIPPED BY USER)"

        self.stdout.write('=' * 60)
        self.stdout.write('BULK FLIGHT RE-ANALYSIS')
        self.stdout.write('=' * 60)
        self.stdout.write(f'Total flights to process: {total_count}')
        self.stdout.write(f'ML Model Status: {ml_status}')
        self.stdout.write(f'Batch size: {batch_size}')
        self.stdout.write(f'Force overwrite: {force_overwrite}')

        if dry_run:
            self.stdout.write(
                self.style.WARNING('\nDRY RUN MODE - No changes will be made')
            )

            # Show sample of what would be processed
            sample_flights = flights[:10]
            for flight in sample_flights:
                status = "✅" if flight.analysis_successful else "❌"
                ml_status = "📊" if hasattr(flight, 'ml_rotation') and flight.ml_rotation else "🤖"
                self.stdout.write(
                    f'{status}{ml_status} Flight {flight.id} - {flight.filename} '
                    f'(Pilot: {flight.pilot.username})'
                )

            if total_count > 10:
                self.stdout.write(f'... and {total_count - 10} more flights')
            return

        # Confirm before proceeding
        if total_count > 20 and not force_overwrite:
            confirm = input(f'\nAre you sure you want to re-analyze {total_count} flights? (y/N): ')
            if confirm.lower() != 'y':
                self.stdout.write('Aborted.')
                return

        # Process in batches
        success_count = 0
        error_count = 0
        skipped_count = 0
        ml_enhanced_count = 0

        for batch_start in range(0, total_count, batch_size):
            batch_end = min(batch_start + batch_size, total_count)
            batch_flights = flights[batch_start:batch_end]

            self.stdout.write(f'\n📦 Processing batch {batch_start//batch_size + 1}: '
                            f'flights {batch_start + 1}-{batch_end}')

            for i, flight in enumerate(batch_flights, batch_start + 1):
                pilot_name = flight.pilot.username if flight.pilot else "No pilot"
                self.stdout.write(
                    f'[{i}/{total_count}] Flight {flight.id} '
                    f'({pilot_name})... ',
                    ending=''
                )

                try:
                    # Check if flight has GPS data
                    gps_data = flight.get_gps_data()
                    if not gps_data:
                        self.stdout.write(
                            self.style.WARNING('SKIPPED - No GPS data')
                        )
                        skipped_count += 1
                        continue

                    # Skip if already successful and not forcing overwrite
                    if flight.analysis_successful and not force_overwrite and not failed_only:
                        self.stdout.write(
                            self.style.WARNING('SKIPPED - Already successful')
                        )
                        skipped_count += 1
                        continue

                    # Reconstruct DataFrame from stored GPS data
                    # Create proper datetime timestamps that FlightManager expects
                    from datetime import datetime, timezone
                    import pandas as pd

                    base_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

                    # Create datetime objects instead of timestamps
                    time_objects = [
                        base_datetime + pd.Timedelta(seconds=i * 0.2)
                        for i in range(len(gps_data))
                    ]

                    df_data = {
                        'time': time_objects,  # Proper datetime objects
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

                    # Store original values for comparison
                    original_rotation = getattr(flight, 'turn_rotation', None)
                    original_ml_rotation = getattr(flight, 'ml_rotation', None)

                    # Re-run the analysis with transaction
                    with transaction.atomic():
                        try:
                            if skip_ml:
                                # Use traditional analysis only
                                self._analyze_traditional_only(manager, flight, df)
                            else:
                                # Use full ML-enhanced analysis
                                manager.analyze_swoop(flight, df)
                        except Exception as e:
                            # If full analysis fails, try traditional analysis as fallback
                            if 'fromisoformat' in str(e) or 'isoformat' in str(e):
                                self.stdout.write('📅', ending='')  # Timestamp issue indicator
                                self._analyze_traditional_only(manager, flight, df)
                            else:
                                raise e

                    # Report results
                    if flight.analysis_successful:
                        if flight.is_swoop:
                            # Show comprehensive ML vs traditional comparison
                            self._display_comprehensive_results(flight, skip_ml, show_detailed_ml)
                            ml_count = getattr(flight, 'ml_predictions_count', 0) or 0
                            if ml_count > 0:
                                ml_enhanced_count += 1
                        else:
                            self.stdout.write(
                                self.style.SUCCESS('NON-SWOOP')
                            )
                        success_count += 1
                    else:
                        self.stdout.write(
                            self.style.ERROR(f'FAILED - {flight.analysis_error}')
                        )
                        error_count += 1

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'ERROR - {str(e)}')
                    )
                    error_count += 1

        # Final summary
        self.stdout.write('\n' + '=' * 60)
        self.stdout.write('BULK RE-ANALYSIS SUMMARY')
        self.stdout.write('=' * 60)
        self.stdout.write(f'Total flights processed: {total_count}')
        self.stdout.write(
            self.style.SUCCESS(f'✅ Successful analyses: {success_count}')
        )
        if ml_enhanced_count > 0:
            self.stdout.write(
                self.style.SUCCESS(f'🤖 ML-enhanced predictions: {ml_enhanced_count}')
            )
        if error_count > 0:
            self.stdout.write(
                self.style.ERROR(f'❌ Failed analyses: {error_count}')
            )
        if skipped_count > 0:
            self.stdout.write(
                self.style.WARNING(f'⏭️  Skipped: {skipped_count}')
            )

        # Calculate success rate
        processed_count = total_count - skipped_count
        if processed_count > 0:
            success_rate = (success_count / processed_count) * 100
            self.stdout.write(f'📊 Success rate: {success_rate:.1f}%')
        elif total_count > 0:
            self.stdout.write('📊 No flights were processed (all skipped)')

        # Overall database statistics
        total_flights = Flight.objects.count()
        successful_flights = Flight.objects.filter(analysis_successful=True).count()
        swoop_flights = Flight.objects.filter(is_swoop=True).count()
        ml_flights = Flight.objects.filter(ml_rotation__isnull=False).count()

        self.stdout.write(f'\n📈 DATABASE OVERVIEW:')
        self.stdout.write(f'   Total flights: {total_flights}')
        self.stdout.write(f'   Successful analyses: {successful_flights} ({successful_flights/total_flights*100:.1f}%)')
        self.stdout.write(f'   Detected swoops: {swoop_flights} ({swoop_flights/total_flights*100:.1f}%)')
        self.stdout.write(f'   ML predictions: {ml_flights} ({ml_flights/total_flights*100:.1f}%)')

    def _analyze_traditional_only(self, manager, flight, df):
        """Run traditional analysis without ML predictions"""
        try:
            # Run basic swoop detection
            landing_idx = manager.get_landing(df)

            try:
                flare_idx = manager.find_flare(df, landing_idx)
                method = "traditional"
            except:
                flare_idx = manager.find_turn_start_fallback(df, landing_idx)
                method = "turn_detection"

            max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

            # Calculate traditional metrics
            turn_rotation = manager.get_rotation(df, flare_idx, max_gspeed_idx)

            # Update flight record
            flight.is_swoop = True
            flight.landing_detected = True
            flight.analysis_successful = True
            flight.analysis_error = ''

            # Store metrics
            flight.turn_rotation = turn_rotation
            flight.turn_direction = "left" if turn_rotation < 0 else "right"
            flight.flare_detection_method = method

            # Store indices
            flight.landing_idx = landing_idx
            flight.flare_idx = flare_idx
            flight.max_vspeed_idx = max_vspeed_idx
            flight.max_gspeed_idx = max_gspeed_idx

            # Store speeds and altitudes
            max_vspeed_ms = abs(df.iloc[max_vspeed_idx]['velD'])
            max_gspeed_ms = df.iloc[max_gspeed_idx]['gspeed']

            flight.max_vertical_speed_ms = max_vspeed_ms
            flight.max_ground_speed_ms = max_gspeed_ms
            flight.max_vertical_speed_mph = max_vspeed_ms * 2.23694
            flight.max_ground_speed_mph = max_gspeed_ms * 2.23694

            flight.flare_altitude_agl = df.iloc[flare_idx]['AGL']
            flight.max_vspeed_altitude_agl = df.iloc[max_vspeed_idx]['AGL']
            flight.max_gspeed_altitude_agl = df.iloc[max_gspeed_idx]['AGL']
            flight.landing_altitude_agl = df.iloc[landing_idx]['AGL']

            flight.analyzed_at = timezone.now()
            flight.save()

        except Exception as e:
            flight.analysis_successful = False
            flight.analysis_error = str(e)
            flight.save()
            raise e

    def _display_comprehensive_results(self, flight, skip_ml, show_detailed=False):
        """Display comprehensive comparison of traditional vs ML predictions"""

        # Basic flight info
        method = getattr(flight, 'flare_detection_method', 'unknown')

        ml_predictions_count = getattr(flight, 'ml_predictions_count', 0) or 0
        if skip_ml or ml_predictions_count == 0:
            # Traditional only
            trad_rotation = getattr(flight, 'turn_rotation', 0) or 0
            self.stdout.write(
                self.style.SUCCESS(f'SWOOP - {trad_rotation:.0f}° ({method})')
            )
            return

        # Show comprehensive ML vs traditional comparison
        comparisons = []

        # 1. Rotation (most important)
        trad_rotation = getattr(flight, 'turn_rotation', 0) or 0
        ml_rotation = getattr(flight, 'ml_rotation', None)
        if ml_rotation is not None:
            rotation_diff = abs(trad_rotation - ml_rotation)
            ml_conf = getattr(flight, 'ml_rotation_confidence', 0) or 0
            comparisons.append(f'Rot: {trad_rotation:.0f}°→{ml_rotation:.0f}° (Δ{rotation_diff:.0f}°, {ml_conf:.2f})')

        # 2. Timing metrics
        timing_metrics = [
            ('turn_time', 'ml_turn_time', 'ml_turn_time_confidence', 'Turn'),
            ('rollout_time', 'ml_rollout_time', 'ml_rollout_time_confidence', 'Rollout'),
            (None, 'ml_swoop_time', 'ml_swoop_time_confidence', 'Swoop')
        ]

        timing_results = []
        for trad_field, ml_field, conf_field, label in timing_metrics:
            ml_value = getattr(flight, ml_field, None)
            if ml_value is not None:
                ml_conf = getattr(flight, conf_field, 0) or 0
                if trad_field and hasattr(flight, trad_field):
                    trad_value = getattr(flight, trad_field, None)
                    if trad_value is not None:
                        diff = abs(trad_value - ml_value)
                        timing_results.append(f'{label}: {trad_value:.1f}s→{ml_value:.1f}s (Δ{diff:.1f}s)')
                    else:
                        timing_results.append(f'{label}: ML={ml_value:.1f}s ({ml_conf:.2f})')
                else:
                    timing_results.append(f'{label}: ML={ml_value:.1f}s ({ml_conf:.2f})')

        if timing_results:
            comparisons.append(f'Time: {", ".join(timing_results)}')

        # 3. Distance/Speed metrics
        distance_metrics = [
            (None, 'ml_distance_to_stop', 'ml_distance_confidence', 'Stop'),
            (None, 'ml_touchdown_distance', 'ml_touchdown_confidence', 'TD'),
            (None, 'ml_touchdown_speed', 'ml_touchdown_speed_confidence', 'TDSpd'),
            (None, 'ml_entry_speed', 'ml_entry_speed_confidence', 'Entry')
        ]

        distance_results = []
        for trad_field, ml_field, conf_field, label in distance_metrics:
            ml_value = getattr(flight, ml_field, None)
            if ml_value is not None:
                ml_conf = getattr(flight, conf_field, 0) or 0
                if 'speed' in ml_field.lower():
                    distance_results.append(f'{label}: {ml_value:.1f}mph ({ml_conf:.2f})')
                else:
                    distance_results.append(f'{label}: {ml_value:.0f}ft ({ml_conf:.2f})')

        if distance_results:
            comparisons.append(f'Dist: {", ".join(distance_results)}')

        # 4. Position metrics (brief)
        position_metrics = [
            ('ml_turn_init_back', 'InitB'),
            ('ml_turn_init_offset', 'InitO')
        ]

        position_results = []
        for ml_field, label in position_metrics:
            ml_value = getattr(flight, ml_field, None)
            if ml_value is not None:
                position_results.append(f'{label}: {ml_value:.0f}ft')

        if position_results:
            comparisons.append(f'Pos: {", ".join(position_results)}')

        # Display results
        ml_count = getattr(flight, 'ml_predictions_count', 0) or 0

        if comparisons:
            result_text = f'SWOOP ({method}, {ml_count} ML)'
            self.stdout.write(self.style.SUCCESS(result_text))

            # Show detailed comparisons with indentation
            for comparison in comparisons:
                self.stdout.write(f'      {comparison}')
        else:
            # Fallback to basic display
            self.stdout.write(
                self.style.SUCCESS(f'SWOOP - {trad_rotation:.0f}° ({method}, {ml_count} ML)')
            )
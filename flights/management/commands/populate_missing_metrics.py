from django.core.management.base import BaseCommand
from flights.models import Flight
from django.db import transaction
from django.db import models

class Command(BaseCommand):
    help = 'Populate missing metrics for existing flights (entry gate speed, rollout altitudes)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of flights to process in each batch',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without making changes',
        )

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        dry_run = options['dry_run']

        # Find swoop flights missing any of the new metrics
        flights_to_update = Flight.objects.filter(
            is_swoop=True,
            analysis_successful=True,
            flare_idx__isnull=False,
            rollout_start_idx__isnull=False,
            rollout_end_idx__isnull=False,
            landing_idx__isnull=False
        ).filter(
            # Missing at least one of these metrics
            models.Q(entry_gate_speed_mph__isnull=True) |
            models.Q(rollout_start_altitude_agl__isnull=True) |
            models.Q(rollout_end_altitude_agl__isnull=True)
        )

        total_flights = flights_to_update.count()
        self.stdout.write(f'Found {total_flights} flights needing metric population')

        if dry_run:
            self.stdout.write('DRY RUN - No changes will be made')
            if total_flights > 0:
                flight = flights_to_update.first()
                self.stdout.write(f'Example flight {flight.id}:')
                metrics = self._calculate_missing_metrics(flight)
                speed_display = f'{metrics["entry_gate_speed"]:.1f}' if metrics["entry_gate_speed"] is not None else 'None'
                start_alt_display = f'{metrics["rollout_start_alt"]:.2f}' if metrics["rollout_start_alt"] is not None else 'None'
                end_alt_display = f'{metrics["rollout_end_alt"]:.2f}' if metrics["rollout_end_alt"] is not None else 'None'

                self.stdout.write(f'  Entry gate speed: {speed_display} mph')
                self.stdout.write(f'  Rollout start altitude: {start_alt_display} m AGL')
                self.stdout.write(f'  Rollout end altitude: {end_alt_display} m AGL')
            return

        updated = 0
        for i in range(0, total_flights, batch_size):
            batch = flights_to_update[i:i + batch_size]

            with transaction.atomic():
                for flight in batch:
                    try:
                        metrics = self._calculate_missing_metrics(flight)
                        updates_made = False

                        # Update entry gate speed if missing
                        if flight.entry_gate_speed_mph is None and metrics['entry_gate_speed'] is not None:
                            flight.entry_gate_speed_mph = metrics['entry_gate_speed']
                            updates_made = True

                        # Update rollout start altitude if missing
                        if flight.rollout_start_altitude_agl is None and metrics['rollout_start_alt'] is not None:
                            flight.rollout_start_altitude_agl = metrics['rollout_start_alt']
                            updates_made = True

                        # Update rollout end altitude if missing
                        if flight.rollout_end_altitude_agl is None and metrics['rollout_end_alt'] is not None:
                            flight.rollout_end_altitude_agl = metrics['rollout_end_alt']
                            updates_made = True

                        if updates_made:
                            flight.save(update_fields=[
                                'entry_gate_speed_mph',
                                'rollout_start_altitude_agl',
                                'rollout_end_altitude_agl'
                            ])
                            updated += 1

                            if updated % 10 == 0:
                                self.stdout.write(f'Updated {updated}/{total_flights} flights...')

                    except Exception as e:
                        self.stdout.write(f'Error updating flight {flight.id}: {e}')

        self.stdout.write(
            self.style.SUCCESS(f'Successfully populated missing metrics for {updated} flights')
        )

    def _calculate_missing_metrics(self, flight):
        """Calculate missing metrics from GPS data"""
        MPH_PER_MPS = 2.23694

        try:
            # Try JSON data first, then GPS points
            gps_data = flight.get_gps_data()

            if gps_data and len(gps_data) > max(flight.flare_idx, flight.rollout_start_idx, flight.rollout_end_idx):
                # Use JSON data
                flare_point = gps_data[flight.flare_idx]
                rollout_start_point = gps_data[flight.rollout_start_idx]
                rollout_end_point = gps_data[flight.rollout_end_idx]

                return {
                    'entry_gate_speed': flare_point.get('ground_speed', 0) * MPH_PER_MPS,
                    'rollout_start_alt': rollout_start_point.get('altitude_agl', 0),
                    'rollout_end_alt': rollout_end_point.get('altitude_agl', 0)
                }
            else:
                # Fallback to GPS points
                gps_points = list(flight.gps_points.order_by('timestamp'))
                max_idx = max(flight.flare_idx, flight.rollout_start_idx, flight.rollout_end_idx)

                if gps_points and len(gps_points) > max_idx:
                    flare_point = gps_points[flight.flare_idx]
                    rollout_start_point = gps_points[flight.rollout_start_idx]
                    rollout_end_point = gps_points[flight.rollout_end_idx]

                    return {
                        'entry_gate_speed': (flare_point.ground_speed or 0) * MPH_PER_MPS,
                        'rollout_start_alt': rollout_start_point.altitude_agl or 0,
                        'rollout_end_alt': rollout_end_point.altitude_agl or 0
                    }

            return {
                'entry_gate_speed': None,
                'rollout_start_alt': None,
                'rollout_end_alt': None
            }

        except Exception as e:
            self.stdout.write(f'Error calculating metrics for flight {flight.id}: {e}')
            return {
                'entry_gate_speed': None,
                'rollout_start_alt': None,
                'rollout_end_alt': None
            }
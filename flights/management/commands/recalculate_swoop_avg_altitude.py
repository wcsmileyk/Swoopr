from django.core.management.base import BaseCommand
from flights.models import Flight
from django.db import transaction
import pandas as pd

class Command(BaseCommand):
    help = 'Recalculate average swoop altitude for existing flights (rollout end to landing)'

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
        # Find swoop flights that need recalculation
        swoops_to_update = Flight.objects.filter(
            is_swoop=True,
            analysis_successful=True,
            rollout_end_idx__isnull=False,
            landing_idx__isnull=False
        )

        total_flights = swoops_to_update.count()
        self.stdout.write(f'Found {total_flights} swoop flights needing average altitude recalculation')

        if dry_run:
            self.stdout.write('DRY RUN - No changes will be made')
            # Show example calculation for first flight
            if total_flights > 0:
                flight = swoops_to_update.first()
                self.stdout.write(f'Example flight {flight.id}:')
                old_avg = flight.swoop_avg_altitude_agl
                new_avg = self._calculate_new_avg_altitude(flight)
                current_display = f'{old_avg:.2f}' if old_avg is not None else 'None'
                new_display = f'{new_avg:.2f}' if new_avg is not None else 'None'
                self.stdout.write(f'  Current avg: {current_display} m AGL')
                self.stdout.write(f'  New avg (rollout end to landing): {new_display} m AGL')
            return

        updated = 0
        for i in range(0, total_flights, batch_size):
            batch = swoops_to_update[i:i + batch_size]

            with transaction.atomic():
                for flight in batch:
                    try:
                        old_avg = flight.swoop_avg_altitude_agl
                        new_avg = self._calculate_new_avg_altitude(flight)

                        if new_avg is not None:
                            flight.swoop_avg_altitude_agl = new_avg
                            flight.save(update_fields=['swoop_avg_altitude_agl'])
                            updated += 1

                            if updated % 10 == 0:
                                self.stdout.write(f'Updated {updated}/{total_flights} flights...')

                    except Exception as e:
                        self.stdout.write(f'Error updating flight {flight.id}: {e}')

        self.stdout.write(
            self.style.SUCCESS(f'Successfully recalculated average altitude for {updated} flights')
        )

    def _calculate_new_avg_altitude(self, flight):
        """Calculate average altitude from rollout end to landing"""
        try:
            # Try JSON data first, then GPS points
            gps_data = flight.get_gps_data()

            if gps_data and len(gps_data) > max(flight.rollout_end_idx, flight.landing_idx):
                # Use JSON data
                rollout_to_landing = gps_data[flight.rollout_end_idx:flight.landing_idx + 1]
                altitudes = [point.get('altitude_agl', 0) for point in rollout_to_landing]
                return sum(altitudes) / len(altitudes) if altitudes else None
            else:
                # Fallback to GPS points
                gps_points = list(flight.gps_points.order_by('timestamp'))
                if gps_points and len(gps_points) > max(flight.rollout_end_idx, flight.landing_idx):
                    rollout_to_landing = gps_points[flight.rollout_end_idx:flight.landing_idx + 1]
                    altitudes = [point.altitude_agl for point in rollout_to_landing if point.altitude_agl is not None]
                    return sum(altitudes) / len(altitudes) if altitudes else None

            return None
        except Exception as e:
            self.stdout.write(f'Error calculating altitude for flight {flight.id}: {e}')
            return None
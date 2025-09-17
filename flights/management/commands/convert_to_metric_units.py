from django.core.management.base import BaseCommand
from flights.models import Flight
from flights.units import *
from django.db import transaction
from django.db import models

class Command(BaseCommand):
    help = 'Convert existing imperial measurements to metric units'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=20,
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

        # Find flights that need metric conversion
        flights_to_convert = Flight.objects.filter(
            models.Q(
                # Has imperial data but missing metric data
                models.Q(max_vertical_speed_mph__isnull=False, max_vertical_speed_ms__isnull=True) |
                models.Q(max_ground_speed_mph__isnull=False, max_ground_speed_ms__isnull=True) |
                models.Q(entry_gate_speed_mph__isnull=False, entry_gate_speed_mps__isnull=True) |
                models.Q(swoop_distance_ft__isnull=False, swoop_distance_m__isnull=True)
            )
        )

        total_flights = flights_to_convert.count()
        self.stdout.write(f'Found {total_flights} flights needing metric conversion')

        if dry_run:
            self.stdout.write('DRY RUN - No changes will be made')
            if total_flights > 0:
                flight = flights_to_convert.first()
                self.stdout.write(f'Example conversion for flight {flight.id}:')

                if flight.max_vertical_speed_mph:
                    metric_vspeed = mph_to_mps(flight.max_vertical_speed_mph)
                    self.stdout.write(f'  Max V-speed: {flight.max_vertical_speed_mph:.1f} mph → {metric_vspeed:.1f} m/s')

                if flight.max_ground_speed_mph:
                    metric_gspeed = mph_to_mps(flight.max_ground_speed_mph)
                    self.stdout.write(f'  Max G-speed: {flight.max_ground_speed_mph:.1f} mph → {metric_gspeed:.1f} m/s')

                if flight.entry_gate_speed_mph:
                    metric_entry = mph_to_mps(flight.entry_gate_speed_mph)
                    self.stdout.write(f'  Entry gate speed: {flight.entry_gate_speed_mph:.1f} mph → {metric_entry:.1f} m/s')

                if flight.swoop_distance_ft:
                    metric_distance = feet_to_meters(flight.swoop_distance_ft)
                    self.stdout.write(f'  Swoop distance: {flight.swoop_distance_ft:.0f} ft → {metric_distance:.1f} m')

            return

        converted = 0
        for i in range(0, total_flights, batch_size):
            batch = flights_to_convert[i:i + batch_size]

            with transaction.atomic():
                for flight in batch:
                    try:
                        updated_fields = []

                        # Convert vertical speed
                        if flight.max_vertical_speed_mph and not flight.max_vertical_speed_ms:
                            flight.max_vertical_speed_ms = mph_to_mps(flight.max_vertical_speed_mph)
                            updated_fields.append('max_vertical_speed_ms')

                        # Convert ground speed
                        if flight.max_ground_speed_mph and not flight.max_ground_speed_ms:
                            flight.max_ground_speed_ms = mph_to_mps(flight.max_ground_speed_mph)
                            updated_fields.append('max_ground_speed_ms')

                        # Convert entry gate speed
                        if flight.entry_gate_speed_mph and not flight.entry_gate_speed_mps:
                            flight.entry_gate_speed_mps = mph_to_mps(flight.entry_gate_speed_mph)
                            updated_fields.append('entry_gate_speed_mps')

                        # Convert swoop distance
                        if flight.swoop_distance_ft and not flight.swoop_distance_m:
                            flight.swoop_distance_m = feet_to_meters(flight.swoop_distance_ft)
                            updated_fields.append('swoop_distance_m')

                        if updated_fields:
                            flight.save(update_fields=updated_fields)
                            converted += 1

                            if converted % 20 == 0:
                                self.stdout.write(f'Converted {converted}/{total_flights} flights...')

                    except Exception as e:
                        self.stdout.write(f'Error converting flight {flight.id}: {e}')

        self.stdout.write(
            self.style.SUCCESS(f'Successfully converted {converted} flights to metric units')
        )

        # Show summary
        self.stdout.write('\n📊 Conversion Summary:')
        metric_complete = Flight.objects.filter(
            max_vertical_speed_ms__isnull=False,
            max_ground_speed_ms__isnull=False,
            swoop_distance_m__isnull=False
        ).count()

        self.stdout.write(f'Flights with complete metric data: {metric_complete}')
        self.stdout.write('Ready for unit-aware templates! 🎉')
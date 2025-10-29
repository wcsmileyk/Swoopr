"""
Management command to calculate gate metrics for flights.

Usage:
    python manage.py calculate_gate_metrics
    python manage.py calculate_gate_metrics --gate-id=1
    python manage.py calculate_gate_metrics --flight-id=123
"""

from django.core.management.base import BaseCommand
from flights.models import Flight, CompetitionGate


class Command(BaseCommand):
    help = 'Calculate competition gate crossing metrics for flights'

    def add_arguments(self, parser):
        parser.add_argument(
            '--gate-id',
            type=int,
            help='Only calculate for flights using this gate ID',
        )
        parser.add_argument(
            '--flight-id',
            type=int,
            help='Only calculate for this specific flight ID',
        )
        parser.add_argument(
            '--recalculate',
            action='store_true',
            help='Recalculate even if gate metrics already exist',
        )

    def handle(self, *args, **options):
        gate_id = options.get('gate_id')
        flight_id = options.get('flight_id')
        recalculate = options.get('recalculate', False)

        # Build queryset
        flights = Flight.objects.filter(competition_gate__isnull=False)

        if flight_id:
            flights = flights.filter(id=flight_id)
        elif gate_id:
            flights = flights.filter(competition_gate_id=gate_id)

        if not recalculate:
            # Only calculate for flights without gate metrics
            flights = flights.filter(gate_crossing_idx__isnull=True)

        total = flights.count()

        if total == 0:
            self.stdout.write(self.style.WARNING('No flights found to process'))
            return

        self.stdout.write(f'Processing {total} flights...')

        success_count = 0
        error_count = 0

        for i, flight in enumerate(flights, 1):
            self.stdout.write(f'[{i}/{total}] Processing flight {flight.id}...', ending='')

            if flight.calculate_gate_metrics():
                success_count += 1
                self.stdout.write(self.style.SUCCESS(' ✓'))
            else:
                error_count += 1
                self.stdout.write(self.style.ERROR(' ✗'))

        self.stdout.write(
            self.style.SUCCESS(f'\nCompleted: {success_count} successful, {error_count} errors')
        )

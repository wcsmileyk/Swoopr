from django.core.management.base import BaseCommand
from flights.models import Flight

class Command(BaseCommand):
    help = 'Populate swoop distances for existing flights'

    def handle(self, *args, **options):
        swoops = Flight.objects.filter(
            is_swoop=True,
            analysis_successful=True,
            swoop_distance_ft__isnull=True
        )

        self.stdout.write(f'Found {swoops.count()} swoops needing distance calculation')

        updated = 0
        for flight in swoops:
            try:
                distance = flight.calculate_and_store_swoop_distance()
                if distance is not None:
                    flight.save()
                    updated += 1
                    if updated % 10 == 0:
                        self.stdout.write(f'Updated {updated} flights...')
            except Exception as e:
                self.stdout.write(f'Error calculating distance for flight {flight.id}: {e}')

        self.stdout.write(
            self.style.SUCCESS(f'Successfully updated {updated} flights with swoop distances')
        )
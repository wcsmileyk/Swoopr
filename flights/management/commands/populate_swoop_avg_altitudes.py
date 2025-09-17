from django.core.management.base import BaseCommand
from flights.models import Flight
import pandas as pd

class Command(BaseCommand):
    help = 'Populate average swoop altitudes for existing flights'

    def handle(self, *args, **options):
        swoops = Flight.objects.filter(
            is_swoop=True,
            analysis_successful=True,
            swoop_avg_altitude_agl__isnull=True,
            flare_idx__isnull=False,
            landing_idx__isnull=False
        )

        self.stdout.write(f'Found {swoops.count()} swoops needing average altitude calculation')

        updated = 0
        for flight in swoops:
            try:
                # Get GPS points for the swoop portion (flare to landing)
                gps_points = flight.gps_points.all().order_by('timestamp')

                if len(gps_points) > max(flight.flare_idx, flight.landing_idx):
                    # Get altitudes from flare to landing
                    swoop_altitudes = []
                    for i in range(flight.flare_idx, flight.landing_idx + 1):
                        if i < len(gps_points):
                            swoop_altitudes.append(gps_points[i].altitude_agl)

                    if swoop_altitudes:
                        flight.swoop_avg_altitude_agl = sum(swoop_altitudes) / len(swoop_altitudes)
                        flight.save()
                        updated += 1

                        if updated % 10 == 0:
                            self.stdout.write(f'Updated {updated} flights...')

            except Exception as e:
                self.stdout.write(f'Error calculating average altitude for flight {flight.id}: {e}')

        self.stdout.write(
            self.style.SUCCESS(f'Successfully updated {updated} flights with average swoop altitudes')
        )
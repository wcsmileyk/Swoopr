from django.core.management.base import BaseCommand
from flights.models import Flight, GPSPoint
from django.db import transaction

class Command(BaseCommand):
    help = 'Migrate existing GPS points to JSON storage'

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

        # Find flights that need migration (have GPS points but no JSON data)
        flights_to_migrate = Flight.objects.filter(
            gps_points__isnull=False,
            gps_data_compressed__isnull=True
        ).distinct()

        total_flights = flights_to_migrate.count()
        self.stdout.write(f'Found {total_flights} flights needing GPS data migration')

        if dry_run:
            self.stdout.write('DRY RUN - No changes will be made')
            return

        migrated = 0
        for i in range(0, total_flights, batch_size):
            batch = flights_to_migrate[i:i + batch_size]

            with transaction.atomic():
                for flight in batch:
                    try:
                        # Get GPS points for this flight
                        gps_points = flight.gps_points.all().order_by('timestamp')

                        if not gps_points.exists():
                            continue

                        # Convert to JSON format
                        gps_data_list = []
                        for point in gps_points:
                            point_data = {
                                'timestamp': point.timestamp.timestamp(),
                                'lat': float(point.location.y),
                                'lon': float(point.location.x),
                                'altitude_msl': float(point.altitude_msl),
                                'altitude_agl': float(point.altitude_agl) if point.altitude_agl else 0,
                                'velocity_north': float(point.velocity_north),
                                'velocity_east': float(point.velocity_east),
                                'velocity_down': float(point.velocity_down),
                                'ground_speed': float(point.ground_speed) if point.ground_speed else 0,
                                'heading': float(point.heading) if point.heading else 0,
                                'h_acc': float(point.horizontal_accuracy) if point.horizontal_accuracy else None,
                                'v_acc': float(point.vertical_accuracy) if point.vertical_accuracy else None,
                                's_acc': float(point.speed_accuracy) if point.speed_accuracy else None,
                                'num_sv': int(point.num_satellites)
                            }
                            gps_data_list.append(point_data)

                        # Store as JSON
                        flight.store_gps_data(gps_data_list)
                        flight.save()

                        migrated += 1
                        if migrated % 10 == 0:
                            self.stdout.write(f'Migrated {migrated}/{total_flights} flights...')

                    except Exception as e:
                        self.stdout.write(f'Error migrating flight {flight.id}: {e}')

        self.stdout.write(
            self.style.SUCCESS(f'Successfully migrated {migrated} flights to JSON storage')
        )

        # Show storage savings
        total_gps_points = GPSPoint.objects.count()
        self.stdout.write(f'Original storage: {total_gps_points:,} individual GPS point records')
        self.stdout.write(f'New storage: {migrated} JSON fields (approx {migrated * 100}KB)')
        savings_ratio = total_gps_points / migrated if migrated > 0 else 0
        self.stdout.write(f'Storage reduction: ~{savings_ratio:.0f}x fewer database records')
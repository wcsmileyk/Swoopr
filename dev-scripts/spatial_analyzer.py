#!/usr/bin/env python3
"""
PostGIS-Enhanced Flight Analyzer
Uses spatial queries for advanced flight path analysis
"""

import os
import django
from django.conf import settings

# Configure Django settings
if not settings.configured:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
    django.setup()

from django.contrib.gis.measure import Distance
from django.contrib.gis.geos import Point, LineString
from flights.models import Flight, GPSPoint
from flights.flysight_manager import save_track_to_database
import numpy as np
from datetime import timedelta


def analyze_flight_spatial(flight):
    """Analyze flight using PostGIS spatial functions"""
    print(f"\n{'='*60}")
    print(f"SPATIAL ANALYSIS: Flight {flight.session_id[:8]}")
    print(f"{'='*60}")

    # Get all GPS points for this flight
    points = flight.gps_points.order_by('timestamp')

    if not points.exists():
        print("No GPS points found for this flight")
        return

    print(f"Device: {flight.device_id}")
    print(f"Points: {points.count()}")

    # Basic stats
    first_point = points.first()
    last_point = points.last()
    duration = last_point.timestamp - first_point.timestamp

    print(f"Duration: {duration}")
    print(f"Start: {first_point.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Altitude analysis
    max_alt_point = points.order_by('-altitude_msl').first()
    min_alt_point = points.order_by('altitude_msl').first()

    print(f"\nALTITUDE ANALYSIS:")
    print(f"Max altitude: {max_alt_point.altitude_msl:.1f}m at {max_alt_point.timestamp.strftime('%H:%M:%S')}")
    print(f"Min altitude: {min_alt_point.altitude_msl:.1f}m at {min_alt_point.timestamp.strftime('%H:%M:%S')}")
    print(f"Total loss: {max_alt_point.altitude_msl - min_alt_point.altitude_msl:.1f}m")

    # Speed analysis using calculated properties
    ground_speeds = [p.ground_speed for p in points]
    total_speeds = [p.total_speed for p in points]

    print(f"\nSPEED ANALYSIS:")
    print(f"Max ground speed: {max(ground_speeds):.1f} m/s ({max(ground_speeds)*2.237:.1f} mph)")
    print(f"Max total speed: {max(total_speeds):.1f} m/s ({max(total_speeds)*2.237:.1f} mph)")
    print(f"Avg ground speed: {np.mean(ground_speeds):.1f} m/s ({np.mean(ground_speeds)*2.237:.1f} mph)")

    # Spatial distance calculation using PostGIS
    total_distance = 0
    prev_point = None

    for point in points[:100]:  # Sample first 100 points for demo
        if prev_point:
            # Use PostGIS distance calculation (more accurate)
            distance = prev_point.location.distance(point.location) * 111000  # Convert degrees to meters approximately
            total_distance += distance
        prev_point = point

    print(f"\nSPATIAL ANALYSIS (first 100 points):")
    print(f"Distance covered: {total_distance:.0f}m")

    # Find flight path bounds
    bounds = points.aggregate(
        min_lat=models.Min('location__y'),
        max_lat=models.Max('location__y'),
        min_lon=models.Min('location__x'),
        max_lon=models.Max('location__x')
    )

    print(f"\nFLIGHT BOUNDS:")
    print(f"Latitude: {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}")
    print(f"Longitude: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}")

    # Detect landing area (points clustered near minimum altitude)
    landing_alt_threshold = min_alt_point.altitude_msl + 50  # Within 50m of lowest point
    landing_points = points.filter(altitude_msl__lte=landing_alt_threshold)

    if landing_points.exists():
        landing_center = landing_points.last()
        print(f"\nLANDING ANALYSIS:")
        print(f"Landing area center: {landing_center.location.y:.6f}, {landing_center.location.x:.6f}")
        print(f"Landing altitude: {landing_center.altitude_msl:.1f}m")

        # Find landing pattern (points within radius of landing)
        landing_radius = 100  # 100 meter radius
        nearby_landing = points.filter(
            location__distance_lte=(landing_center.location, Distance(m=landing_radius))
        )
        print(f"Points within {landing_radius}m of landing: {nearby_landing.count()}")

    # High-speed segments
    high_speed_threshold = 50  # m/s
    high_speed_points = [p for p in points if p.ground_speed > high_speed_threshold]

    if high_speed_points:
        print(f"\nHIGH-SPEED ANALYSIS:")
        print(f"Points above {high_speed_threshold} m/s: {len(high_speed_points)}")

        # Find peak speed moment
        peak_speed_point = max(high_speed_points, key=lambda p: p.total_speed)
        print(f"Peak speed: {peak_speed_point.total_speed:.1f} m/s at {peak_speed_point.timestamp.strftime('%H:%M:%S')}")
        print(f"Peak speed altitude: {peak_speed_point.altitude_msl:.1f}m")


def load_and_analyze_sample_files():
    """Load sample files into database and analyze them"""
    print("Loading sample files into database...")

    sample_files = [
        'sample_tracks/25-09-12-sw5.csv',
        'sample_tracks/25-09-15-sw4.CSV'
    ]

    flights = []
    for file_path in sample_files:
        if os.path.exists(file_path):
            try:
                flight = save_track_to_database(file_path)
                flights.append(flight)
                print(f"Loaded {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    # Analyze each flight
    for flight in flights:
        analyze_flight_spatial(flight)

    return flights


def spatial_queries_demo():
    """Demonstrate PostGIS spatial queries"""
    print(f"\n{'='*60}")
    print("POSTGIS SPATIAL QUERIES DEMO")
    print(f"{'='*60}")

    # Get all flights
    flights = Flight.objects.all()

    if not flights.exists():
        print("No flights in database. Run load_and_analyze_sample_files() first.")
        return

    # Cross-flight analysis
    total_points = GPSPoint.objects.count()
    print(f"Total GPS points in database: {total_points}")

    # Find all high-altitude points across all flights
    high_altitude_points = GPSPoint.objects.filter(altitude_msl__gte=4000)
    print(f"Points above 4000m: {high_altitude_points.count()}")

    # Find all high-speed points
    high_speed_points = GPSPoint.objects.extra(
        where=["sqrt(velocity_north*velocity_north + velocity_east*velocity_east) > %s"],
        params=[60]
    )
    print(f"Points with ground speed > 60 m/s: {high_speed_points.count()}")

    # Spatial clustering example (if you have multiple flights)
    if flights.count() > 1:
        print("\nSPATIAL CLUSTERING ANALYSIS:")
        # Find points within 1km of each other across different flights
        reference_point = GPSPoint.objects.first()
        if reference_point:
            nearby_points = GPSPoint.objects.filter(
                location__distance_lte=(reference_point.location, Distance(km=1))
            ).exclude(flight=reference_point.flight)
            print(f"Points within 1km of reference across different flights: {nearby_points.count()}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        spatial_queries_demo()
    else:
        flights = load_and_analyze_sample_files()
        if flights:
            spatial_queries_demo()
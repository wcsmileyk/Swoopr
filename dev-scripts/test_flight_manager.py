#!/usr/bin/env python3
"""
Test the consolidated flight manager
"""

import os
import sys
import django

# Setup Django
sys.path.append('/home/smiley/PycharmProjects/Swoopr')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import process_flysight_file
from flights.models import Flight, GPSPoint


def test_flight_manager():
    """Test the consolidated flight manager"""

    # Test file
    test_file = 'sample_tracks/25-09-12-sw5.csv'

    print(f"Testing consolidated flight manager with: {test_file}")
    print("=" * 60)

    try:
        # Process the file
        flight = process_flysight_file(test_file)

        print(f"✅ Flight created successfully:")
        print(f"   ID: {flight.id}")
        print(f"   Device: {flight.device_id}")
        print(f"   Session: {flight.session_id}")
        print(f"   Is Swoop: {flight.is_swoop}")
        print(f"   Analysis Successful: {flight.analysis_successful}")

        if flight.analysis_successful:
            print(f"\n📊 Swoop Analysis Results:")
            print(f"   Turn Rotation: {flight.turn_rotation:.1f}° ({flight.turn_direction})")
            print(f"   Max Vertical Speed: {flight.max_vertical_speed_mph:.1f} mph")
            print(f"   Max Ground Speed: {flight.max_ground_speed_mph:.1f} mph")
            print(f"   Turn Time: {flight.turn_time:.1f} sec")
            print(f"   Rollout Time: {flight.rollout_time:.1f} sec")
            print(f"   Exit Altitude: {flight.exit_altitude_agl:.0f}m AGL")
            print(f"   Flare Altitude: {flight.flare_altitude_agl:.0f}m AGL")

        # Check GPS points
        gps_count = flight.gps_points.count()
        print(f"\n📍 GPS Points: {gps_count} records")

        if gps_count > 0:
            first_point = flight.gps_points.first()
            last_point = flight.gps_points.last()
            print(f"   First: {first_point.timestamp} at {first_point.location}")
            print(f"   Last:  {last_point.timestamp} at {last_point.location}")

        # Compare with gswoop if available
        try:
            import subprocess
            result = subprocess.run(['gswoop', '-i', test_file],
                                  capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'degrees of rotation:' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            gswoop_rotation = float(parts[3])
                            gswoop_direction = parts[5].replace('(', '').replace('-hand)', '')

                            print(f"\n🔄 Comparison with gswoop:")
                            print(f"   gswoop:        {gswoop_rotation}° ({gswoop_direction})")
                            print(f"   flight_manager: {abs(flight.turn_rotation):.1f}° ({flight.turn_direction})")
                            print(f"   Difference:    {abs(abs(flight.turn_rotation) - gswoop_rotation):.1f}°")
                            break
            else:
                print(f"\n⚠️  gswoop failed: {result.stderr}")

        except Exception as e:
            print(f"\n⚠️  Could not run gswoop comparison: {e}")

        return flight

    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    flight = test_flight_manager()
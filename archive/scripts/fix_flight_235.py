#!/usr/bin/env python3
"""
Fix GPS data storage for flight 235
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.models import Flight
from flights.flight_manager import FlightManager
import pandas as pd

def fix_flight_235():
    """Fix GPS data storage for flight 235"""

    flight = Flight.objects.get(id=235)
    print(f"Flight {flight.id}: {flight.filename}")
    print(f"Pilot: {flight.pilot}")
    print(f"Current GPS points: {flight.gps_points.count()}")

    # Check if this is the temp file from /home/smiley/Downloads/
    temp_file_path = f"/tmp/{flight.filename}"
    downloads_file_path = f"/home/smiley/Downloads/{flight.filename}"

    file_to_use = None
    if os.path.exists(temp_file_path):
        file_to_use = temp_file_path
        print(f"✅ Found temp file: {temp_file_path}")
    elif os.path.exists(downloads_file_path):
        file_to_use = downloads_file_path
        print(f"✅ Found downloads file: {downloads_file_path}")
    else:
        print(f"❌ File not found in temp or downloads")
        return False

    try:
        # Read the file and store GPS data
        manager = FlightManager()
        df, metadata = manager.read_flysight_file(file_to_use)
        print(f"File contains {len(df)} GPS points")

        # Convert to GPS data list format
        gps_data_list = []
        for idx, row in df.iterrows():
            gps_data_list.append({
                'timestamp': pd.Timestamp(row['time']),
                'latitude': row['lat'],
                'longitude': row['lon'],
                'altitude': row['hMSL'],
                'altitude_agl': row['AGL'],
                'vertical_speed': row['velD'],
                'ground_speed': row['gspeed'],
                'north_velocity': row['velN'],
                'east_velocity': row['velE']
            })

        # Store GPS data
        flight.store_gps_data(gps_data_list)
        print(f"✅ Stored {flight.gps_points.count()} GPS points")
        print(f"✅ Flight {flight.id} chart should now render properly")
        return True

    except Exception as e:
        print(f"❌ Error fixing flight: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fix_flight_235()
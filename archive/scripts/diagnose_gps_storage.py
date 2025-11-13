#!/usr/bin/env python3
"""
Diagnose GPS storage issues during flight upload
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager
from flights.models import Flight
from users.models import User

def diagnose_gps_storage():
    """Diagnose GPS storage issues"""

    print("🔍 DIAGNOSING GPS STORAGE ISSUES")
    print("=" * 50)

    # Use the problematic file from before
    test_file = "/home/smiley/Downloads/gps_00545.csv"

    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return

    user = User.objects.first()
    print(f"Using pilot: {user}")

    # Get initial flight count
    initial_count = Flight.objects.count()
    print(f"Initial flight count: {initial_count}")

    try:
        # Test the full upload process
        manager = FlightManager()
        print(f"\n📁 Processing file: {test_file}")

        # Step 1: Read file
        print("Step 1: Reading file...")
        df, metadata = manager.read_flysight_file(test_file)
        print(f"✅ Read {len(df)} data points")

        # Step 2: Create flight (this might reuse existing flight)
        print("Step 2: Creating/updating flight...")
        flight = manager.create_or_update_flight(test_file, metadata, user)
        print(f"✅ Flight: {flight.id} ({flight.filename})")

        # Step 3: Check GPS data before create_gps_points
        has_gps_before = bool(flight.gps_data_compressed)
        print(f"GPS data before create_gps_points: {'Yes' if has_gps_before else 'No'}")

        # Step 4: Create GPS points (with detailed logging)
        print("Step 3: Creating GPS points...")
        try:
            manager.create_gps_points(flight, df)
            flight.refresh_from_db()  # Refresh to get latest data
            has_gps_after = bool(flight.gps_data_compressed)
            print(f"✅ GPS data after create_gps_points: {'Yes' if has_gps_after else 'No'}")

            # Test GPS data retrieval
            if has_gps_after:
                gps_data = flight.get_gps_data()
                if gps_data:
                    print(f"✅ GPS data retrieval test: {len(gps_data)} points")
                else:
                    print(f"❌ GPS data retrieval failed")
        except Exception as e:
            print(f"❌ Error in create_gps_points: {e}")
            import traceback
            traceback.print_exc()

        # Step 5: Analyze swoop
        print("Step 4: Analyzing swoop...")
        try:
            manager.analyze_swoop(flight, df)
            print(f"✅ Analysis complete")
        except Exception as e:
            print(f"❌ Error in analyze_swoop: {e}")
            import traceback
            traceback.print_exc()

        # Final status
        flight.refresh_from_db()
        has_final_gps = bool(flight.gps_data_compressed)
        print(f"\n📊 FINAL RESULTS:")
        print(f"Flight ID: {flight.id}")
        print(f"GPS data stored: {'Yes' if has_final_gps else 'No'}")
        print(f"Analysis successful: {flight.analysis_successful}")
        print(f"Landing detected: {flight.landing_detected}")

        if has_final_gps:
            gps_data = flight.get_gps_data()
            if gps_data:
                print(f"✅ GPS storage successful - {len(gps_data)} points - Chart should render")
            else:
                print(f"❌ GPS data corrupted - Chart will not render")
        else:
            print(f"❌ GPS STORAGE FAILED - Chart will not render")

    except Exception as e:
        print(f"❌ Overall error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_gps_storage()
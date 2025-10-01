#!/usr/bin/env python3
"""
Test ML integration in FlightManager
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager
from users.models import User

def test_ml_integration():
    """Test the integrated ML system"""

    # Get a test user
    user = User.objects.first()
    print(f"Using user: {user}")

    # Test the flight processing with integrated ML
    manager = FlightManager()
    flight_path = "/home/smiley/Downloads/gps_00545.csv"

    try:
        print(f"Processing flight: {flight_path}")
        flight = manager.process_file(flight_path, pilot=user)
        print(f"Flight processed successfully: {flight.id}")

        # Check basic ML predictions
        print(f"\n=== Basic ML Rotation Predictions ===")
        print(f"ML Rotation: {flight.ml_rotation}°")
        print(f"ML Confidence: {flight.ml_rotation_confidence}")
        print(f"ML Method: {flight.ml_rotation_method}")
        print(f"ML Intended turn: {flight.ml_intended_turn}")

        # Check multi-metric ML predictions
        print(f"\n=== Multi-metric ML Predictions ===")
        print(f"ML Turn time: {flight.ml_turn_time}")
        print(f"ML Rollout time: {flight.ml_rollout_time}")
        print(f"ML Swoop time: {flight.ml_swoop_time}")
        print(f"ML Distance to stop: {flight.ml_distance_to_stop}")
        print(f"ML Touchdown distance: {flight.ml_touchdown_distance}")
        print(f"ML Touchdown speed: {flight.ml_touchdown_speed}")
        print(f"ML Entry speed: {flight.ml_entry_speed}")
        print(f"ML Turn init back: {flight.ml_turn_init_back}")
        print(f"ML Turn init offset: {flight.ml_turn_init_offset}")
        print(f"ML Predictions count: {flight.ml_predictions_count}")
        print(f"ML Predictions updated: {flight.ml_predictions_updated_at}")

        # Check traditional metrics for comparison
        print(f"\n=== Traditional Metrics for Comparison ===")
        print(f"Traditional rotation: {flight.turn_rotation}°")
        print(f"Traditional confidence: {flight.turn_rotation_confidence}")

        print(f"\n✅ ML integration test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error processing flight: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ml_integration()
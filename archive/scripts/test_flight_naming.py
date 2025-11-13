#!/usr/bin/env python3
"""
Test the flight naming system
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.models import Flight
from users.models import User

def test_flight_naming():
    """Test the flight naming system"""

    print("🏷️  TESTING FLIGHT NAMING SYSTEM")
    print("=" * 50)

    user = User.objects.first()
    print(f"Testing with user: {user}")

    # Test 1: Check existing flights and their naming
    existing_flights = Flight.objects.filter(pilot=user).order_by('created_at')
    print(f"\n📋 Existing flights for {user}:")

    for flight in existing_flights[:5]:  # Show first 5
        flight_datetime = flight.get_flight_datetime()
        current_name = flight.flight_name or "(no name)"
        print(f"  Flight {flight.id}: {current_name}")
        print(f"    GPS datetime: {flight_datetime}")
        print(f"    Date: {flight_datetime.date()}")
        print()

    # Test 2: Update one flight's name to see the system work
    if existing_flights.exists():
        test_flight = existing_flights.first()
        print(f"🔄 Testing name generation for flight {test_flight.id}")

        # Generate chronological name
        generated_name = test_flight.generate_chronological_name()
        print(f"  Generated name: {generated_name}")

        # Update the name
        test_flight.update_chronological_name()
        print(f"  Updated flight name: {test_flight.flight_name}")

        # Test resequencing for the pilot's flights on this date
        flight_datetime = test_flight.get_flight_datetime()
        flight_date = flight_datetime.date()
        print(f"  Resequencing all flights for {user} on {flight_date}")

        Flight.resequence_pilot_flights(user, target_date=flight_date)

        # Show updated names
        same_day_flights = []
        for flight in Flight.objects.filter(pilot=user):
            if flight.get_flight_datetime().date() == flight_date:
                same_day_flights.append((flight, flight.get_flight_datetime()))

        same_day_flights.sort(key=lambda x: x[1])
        print(f"  Flights on {flight_date} after resequencing:")
        for i, (flight, dt) in enumerate(same_day_flights, 1):
            print(f"    {i}. {flight.flight_name} (GPS time: {dt.strftime('%H:%M:%S')})")

    print(f"\n✅ FLIGHT NAMING SYSTEM TEST COMPLETE")

if __name__ == "__main__":
    test_flight_naming()
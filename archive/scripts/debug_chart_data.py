#!/usr/bin/env python3
"""
Debug chart data for flight 235
"""

import os
import sys
import django
import json
import math

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.models import Flight

def debug_chart_data():
    """Debug chart data structure and content"""

    flight = Flight.objects.get(id=235)
    chart_data = flight.get_chart_data()

    print("=== CHART DATA ANALYSIS ===")
    print(f"Chart data exists: {bool(chart_data)}")

    if not chart_data:
        print("❌ No chart data - this is the problem!")
        return

    print(f"Data arrays:")
    for key in ['timestamps', 'altitude_agl', 'vertical_speed', 'ground_speed']:
        if key in chart_data:
            values = chart_data[key]
            print(f"  {key}: {len(values)} values")

            # Check for problematic values
            if key != 'timestamps':
                try:
                    invalid_count = sum(1 for x in values if x is None or (isinstance(x, (int, float)) and math.isnan(x)))
                    print(f"    Invalid values: {invalid_count}")
                    if len(values) > 0:
                        print(f"    Sample values: {values[:3]}")
                except Exception as e:
                    print(f"    Error checking values: {e}")

    # Check important points
    important_points = chart_data.get('important_points', {})
    print(f"\nImportant points:")
    for key, value in important_points.items():
        print(f"  {key}: {value} ({type(value).__name__})")

    # Try to JSON serialize the data (this is what the template does)
    try:
        json_str = json.dumps(chart_data)
        print(f"\n✅ JSON serialization successful ({len(json_str)} chars)")
    except Exception as e:
        print(f"\n❌ JSON serialization failed: {e}")

        # Find the problematic data
        for key, value in chart_data.items():
            try:
                json.dumps({key: value})
                print(f"  {key}: OK")
            except Exception as sub_e:
                print(f"  {key}: ❌ {sub_e}")

if __name__ == "__main__":
    debug_chart_data()
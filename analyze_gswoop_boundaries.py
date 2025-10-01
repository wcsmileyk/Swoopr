#!/usr/bin/env python3
"""
Analyze gswoop's turn boundary detection to understand methodology differences
"""

import os
import sys
import django
import subprocess
import re
import numpy as np

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

def parse_gswoop_detailed(output):
    """Parse detailed gswoop output for all metrics"""
    data = {}

    patterns = {
        'initiated_turn_alt': r'initiated turn:\s+(\d+)\s+ft AGL',
        'initiated_turn_back': r'initiated turn:\s+\d+\s+ft AGL,\s+(\d+)\s+ft back',
        'max_vspeed_alt': r'max vertical speed:\s+(\d+)\s+ft AGL',
        'max_vspeed_back': r'max vertical speed:.*,\s+(\d+)\s+ft back',
        'max_vspeed_mph': r'max vertical speed:.*\((\d+\.\d+)\s+mph\)',
        'rotation_deg': r'degrees of rotation:\s+(\d+)\s+deg',
        'rotation_dir': r'degrees of rotation:.*\((\w+)-hand\)',
        'turn_time': r'time to execute turn:\s+(\d+\.\d+)\s+sec',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            try:
                data[key] = float(match.group(1))
            except:
                data[key] = match.group(1)

    return data

def analyze_turn_boundaries(filepath):
    """Analyze turn boundaries for gswoop vs our algorithm"""

    print(f"🔍 Analyzing turn boundaries: {os.path.basename(filepath)}")
    print("=" * 60)

    # Get gswoop analysis
    result = subprocess.run(['gswoop', '-i', filepath],
                          capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print("❌ gswoop failed")
        return

    gswoop_data = parse_gswoop_detailed(result.stdout)
    print("📊 gswoop Analysis:")
    print(f"   Turn initiated: {gswoop_data.get('initiated_turn_alt', 'N/A')} ft AGL, {gswoop_data.get('initiated_turn_back', 'N/A')} ft back")
    print(f"   Max vertical speed: {gswoop_data.get('max_vspeed_alt', 'N/A')} ft AGL, {gswoop_data.get('max_vspeed_back', 'N/A')} ft back")
    print(f"   Rotation: {gswoop_data.get('rotation_deg', 'N/A')}° ({gswoop_data.get('rotation_dir', 'N/A')}-hand)")
    print(f"   Turn time: {gswoop_data.get('turn_time', 'N/A')} sec")

    # Analyze with our algorithm
    manager = FlightManager()
    df, metadata = manager.read_flysight_file(filepath)

    landing_idx = manager.get_landing(df)

    try:
        flare_idx = manager.find_flare(df, landing_idx)
        flare_method = "traditional"
    except:
        flare_idx = manager.find_turn_start_fallback(df, landing_idx)
        flare_method = "fallback"

    max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

    print(f"\n🔄 Our Algorithm:")
    print(f"   Flare detection: {flare_method}")
    print(f"   Turn start (flare): {df.iloc[flare_idx]['AGL']:.0f} ft AGL (idx: {flare_idx})")
    print(f"   Max vspeed: {df.iloc[max_vspeed_idx]['AGL']:.0f} ft AGL (idx: {max_vspeed_idx})")
    print(f"   Max gspeed: {df.iloc[max_gspeed_idx]['AGL']:.0f} ft AGL (idx: {max_gspeed_idx})")
    print(f"   Turn end (max gspeed): {df.iloc[max_gspeed_idx]['AGL']:.0f} ft AGL")

    # Calculate turn segment using our boundaries
    our_turn_data = df[flare_idx:max_gspeed_idx+1]
    our_turn_time = len(our_turn_data) * 0.2

    print(f"   Our turn time: {our_turn_time:.1f} sec")
    print(f"   Our turn points: {len(our_turn_data)}")

    # Heading analysis for our segment
    headings = our_turn_data['heading'].values
    start_heading = headings[0]
    end_heading = headings[-1]
    net_change = end_heading - start_heading

    while net_change > 180:
        net_change -= 360
    while net_change < -180:
        net_change += 360

    print(f"   Start heading: {start_heading:.1f}°")
    print(f"   End heading: {end_heading:.1f}°")
    print(f"   Net heading change: {net_change:.1f}°")

    # Try to estimate gswoop's turn boundaries
    print(f"\n🔍 Boundary Analysis:")

    # gswoop seems to use "back" distance from landing
    # Let's try to find points that match gswoop's distances

    if 'initiated_turn_back' in gswoop_data and 'max_vspeed_back' in gswoop_data:
        # Try to find indices that match gswoop's "back" distances
        # This is complex without 3D position data, but we can estimate

        gswoop_turn_back = gswoop_data['initiated_turn_back']
        gswoop_vspeed_back = gswoop_data['max_vspeed_back']

        print(f"   gswoop turn initiation: {gswoop_turn_back} ft back from landing")
        print(f"   gswoop max vspeed: {gswoop_vspeed_back} ft back from landing")

        # Estimate what this means in terms of time/indices
        # FlySight at 5Hz = 0.2s per point, typical swoop ~50mph = ~73 ft/s
        # So roughly 73 ft/s * 0.2s = ~15 ft per GPS point

        estimated_turn_points_back = gswoop_turn_back / 15
        estimated_vspeed_points_back = gswoop_vspeed_back / 15

        print(f"   Estimated turn start: ~{estimated_turn_points_back:.0f} points back from landing")
        print(f"   Estimated max vspeed: ~{estimated_vspeed_points_back:.0f} points back from landing")

        # Compare with our indices
        our_flare_back = landing_idx - flare_idx
        our_vspeed_back = landing_idx - max_vspeed_idx
        our_gspeed_back = landing_idx - max_gspeed_idx

        print(f"   Our flare: {our_flare_back} points back from landing")
        print(f"   Our max vspeed: {our_vspeed_back} points back from landing")
        print(f"   Our max gspeed: {our_gspeed_back} points back from landing")

        # Check if we're using different boundaries
        flare_diff = abs(our_flare_back - estimated_turn_points_back)
        vspeed_diff = abs(our_vspeed_back - estimated_vspeed_points_back)

        print(f"\n⚠️  Boundary Differences:")
        print(f"   Turn start difference: ~{flare_diff:.0f} points ({flare_diff * 15:.0f} ft)")
        print(f"   Max vspeed difference: ~{vspeed_diff:.0f} points ({vspeed_diff * 15:.0f} ft)")

        if flare_diff > 5:
            print(f"   🚨 SIGNIFICANT TURN START DIFFERENCE")
        if vspeed_diff > 5:
            print(f"   🚨 SIGNIFICANT MAX VSPEED DIFFERENCE")

def analyze_problematic_cases():
    """Analyze the cases where expert confirmed gswoop was correct"""

    test_cases = [
        "~/FlySight/Training/25-02-20/25-02-20-sw1.csv",  # Expert: left 270
        "~/FlySight/Training/25-03-27/25-03-27-sw4.csv",  # Expert: left 270
    ]

    for filepath in test_cases:
        filepath = os.path.expanduser(filepath)
        analyze_turn_boundaries(filepath)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    analyze_problematic_cases()
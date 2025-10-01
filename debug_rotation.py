#!/usr/bin/env python3
"""
Debug rotation calculation differences between our algorithm and gswoop
"""

import os
import sys
import django
import subprocess
import re

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

def parse_gswoop_rotation(output):
    """Parse rotation from gswoop output"""
    rotation_match = re.search(r'degrees of rotation:\s+(\d+)\s+deg\s+\((\w+)-hand\)', output)
    if rotation_match:
        degrees = float(rotation_match.group(1))
        direction = rotation_match.group(2)
        return degrees if direction == 'right' else -degrees
    return None

def debug_file(filepath):
    """Debug rotation calculation for a specific file"""
    print(f"🔍 Debugging: {filepath}")
    print("=" * 60)

    # Get gswoop result
    result = subprocess.run(['gswoop', '-i', filepath],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ gswoop failed")
        return

    gswoop_rotation = parse_gswoop_rotation(result.stdout)
    print(f"📊 gswoop rotation: {gswoop_rotation}°")

    # Analyze with our algorithm
    manager = FlightManager()
    df, metadata = manager.read_flysight_file(filepath)

    landing_idx = manager.get_landing(df)
    try:
        flare_idx = manager.find_flare(df, landing_idx)
    except:
        flare_idx = manager.find_turn_start_fallback(df, landing_idx)

    max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

    # Get detailed rotation analysis
    our_rotation, intended_turn, confidence, method = manager.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)

    print(f"🔄 Our rotation: {our_rotation}°")
    print(f"🎯 Intended turn: {intended_turn}°")
    print(f"📈 Confidence: {confidence:.2f}")
    print(f"🔧 Method: {method}")

    # Analyze turn segment in detail
    turn_data = df[flare_idx:max_gspeed_idx+1]
    headings = turn_data['heading'].values

    print(f"\n📐 Heading Analysis:")
    print(f"   Start heading: {headings[0]:.1f}°")
    print(f"   End heading: {headings[-1]:.1f}°")
    print(f"   Net change: {headings[-1] - headings[0]:.1f}°")
    print(f"   Turn points: {len(headings)}")

    # Manual calculation
    total_change = 0
    direction_changes = 0
    last_diff = None

    for i in range(1, len(headings)):
        diff = headings[i] - headings[i-1]

        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        total_change += abs(diff)

        # Track direction changes
        if last_diff is not None:
            if (last_diff > 0) != (diff > 0):
                direction_changes += 1
        last_diff = diff

    print(f"   Total heading change: {total_change:.1f}°")
    print(f"   Direction changes: {direction_changes}")

    # Check for full rotations
    net_change = headings[-1] - headings[0]
    while net_change > 180:
        net_change -= 360
    while net_change < -180:
        net_change += 360

    estimated_full_rotations = max(0, int((total_change - abs(net_change)) / 360))
    total_with_rotations = abs(net_change) + (estimated_full_rotations * 360)

    print(f"   Net change: {net_change:.1f}°")
    print(f"   Estimated full rotations: {estimated_full_rotations}")
    print(f"   Total with full rotations: {total_with_rotations:.1f}°")

    difference = abs(our_rotation - gswoop_rotation) if gswoop_rotation else 0
    print(f"\n❗ Difference: {difference:.1f}°")

    if difference > 50:
        print("🚨 LARGE DIFFERENCE - Possible issues:")
        print("   - Direction interpretation (left vs right)")
        print("   - Full rotation counting")
        print("   - Turn start/end point detection")
        print("   - Algorithm implementation differences")

if __name__ == "__main__":
    # Debug a few specific files
    test_files = [
        "~/FlySight/Training/24-10-12/24-10-12-sw3.csv",  # Good agreement
        "~/FlySight/Training/25-07-04/25-07-04-sw3.csv",  # Good agreement
        "~/FlySight/Training/25-04-10/25-04-10-sw5.csv",  # Large difference
        "~/FlySight/Training/25-02-22/25-02-22-sw5.csv"   # Large difference
    ]

    for filepath in test_files:
        debug_file(os.path.expanduser(filepath))
        print("\n" + "="*80 + "\n")
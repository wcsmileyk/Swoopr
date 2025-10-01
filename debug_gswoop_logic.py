#!/usr/bin/env python3
"""
Debug gswoop's logic more carefully by examining the flight path around boundary points
"""

import os
import sys
import django
import subprocess
import re
import numpy as np
import pandas as pd

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

def debug_gswoop_boundaries(filepath):
    """Debug gswoop boundaries by examining flight data around key points"""

    print(f"🔍 DEBUGGING GSWOOP LOGIC: {os.path.basename(filepath)}")
    print("=" * 70)

    # Get gswoop analysis
    result = subprocess.run(['gswoop', '-i', filepath],
                          capture_output=True, text=True, timeout=30)

    print("📋 Full gswoop output:")
    print(result.stdout)
    print("-" * 50)

    # Parse key altitudes
    patterns = {
        'turn_start': r'initiated turn:\s+(\d+)\s+ft AGL',
        'max_vspeed': r'max vertical speed:\s+(\d+)\s+ft AGL',
        'rollout_start': r'started rollout:\s+(\d+)\s+ft AGL',
        'rollout_end': r'finished rollout:\s+(\d+)\s+ft AGL',
    }

    altitudes = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, result.stdout)
        if match:
            altitudes[key] = float(match.group(1))

    # Load flight data
    manager = FlightManager()
    df, metadata = manager.read_flysight_file(filepath)

    print(f"📊 Flight Data Overview:")
    print(f"   Total GPS points: {len(df)}")
    print(f"   Altitude range: {df['AGL'].min()/0.3048:.0f} - {df['AGL'].max()/0.3048:.0f} ft AGL")
    print(f"   Flight duration: {len(df) * 0.2:.1f} seconds")

    # Find points around each gswoop altitude marker
    for phase, target_alt_ft in altitudes.items():
        print(f"\n🎯 Analyzing {phase}: {target_alt_ft} ft AGL")

        target_alt_m = target_alt_ft * 0.3048

        # Find all points within ±20ft of target altitude
        altitude_diffs = np.abs(df['AGL'] - target_alt_m)
        close_points = df[altitude_diffs <= (20 * 0.3048)].copy()

        if len(close_points) > 0:
            # Sort by altitude difference
            close_points['alt_diff_ft'] = np.abs(close_points['AGL'] - target_alt_m) / 0.3048
            close_points = close_points.sort_values('alt_diff_ft')

            print(f"   Found {len(close_points)} points within ±20ft:")

            # Show top 5 closest matches
            for i, (idx, row) in enumerate(close_points.head().iterrows()):
                alt_ft = row['AGL'] / 0.3048
                vspeed_mph = abs(row['velD']) * 2.23694
                gspeed_mph = row['gspeed'] * 2.23694
                heading = row['heading']

                print(f"     #{i+1}: idx {idx}, {alt_ft:.1f}ft (±{row['alt_diff_ft']:.1f}ft), "
                      f"vspeed: {vspeed_mph:.1f}mph, gspeed: {gspeed_mph:.1f}mph, hdg: {heading:.1f}°")

            # Check if gswoop might be looking for specific criteria beyond just altitude
            closest_idx = close_points.index[0]
            closest_row = close_points.iloc[0]

            print(f"   Best match characteristics:")
            print(f"     Vertical speed: {abs(closest_row['velD']) * 2.23694:.1f} mph")
            print(f"     Ground speed: {closest_row['gspeed'] * 2.23694:.1f} mph")
            print(f"     Heading: {closest_row['heading']:.1f}°")

            # Look at trend around this point
            window_start = max(0, closest_idx - 10)
            window_end = min(len(df), closest_idx + 11)
            window = df.iloc[window_start:window_end]

            print(f"   Context (±10 points around match):")

            vspeed_trend = np.diff(np.abs(window['velD']) * 2.23694)
            gspeed_trend = np.diff(window['gspeed'] * 2.23694)
            heading_changes = np.diff(window['heading'])

            # Normalize heading changes
            heading_changes = np.where(heading_changes > 180, heading_changes - 360, heading_changes)
            heading_changes = np.where(heading_changes < -180, heading_changes + 360, heading_changes)

            avg_vspeed_change = np.mean(vspeed_trend) if len(vspeed_trend) > 0 else 0
            avg_gspeed_change = np.mean(gspeed_trend) if len(gspeed_trend) > 0 else 0
            avg_heading_change = np.mean(np.abs(heading_changes)) if len(heading_changes) > 0 else 0

            print(f"     Avg vspeed change: {avg_vspeed_change:+.1f} mph/point")
            print(f"     Avg gspeed change: {avg_gspeed_change:+.1f} mph/point")
            print(f"     Avg heading change: {avg_heading_change:.1f}°/point")

            # Analyze what gswoop might be detecting
            if phase == 'turn_start':
                if avg_heading_change > 2:
                    print(f"     💡 High heading change rate - likely start of turn")
                if avg_vspeed_change > 0.5:
                    print(f"     💡 Increasing vertical speed - turn initiation")

            elif phase == 'max_vspeed':
                print(f"     💡 Should be point of maximum vertical speed")

            elif phase == 'rollout_start':
                if avg_heading_change < 2:
                    print(f"     💡 Low heading change - end of turn/start of rollout")
                if avg_vspeed_change < -0.5:
                    print(f"     💡 Decreasing vertical speed - flare/rollout")

        else:
            print(f"   ❌ No points found within ±20ft of {target_alt_ft}ft")

    # Try to understand the turn segment gswoop is measuring
    if 'turn_start' in altitudes and 'max_vspeed' in altitudes:
        turn_start_alt = altitudes['turn_start'] * 0.3048
        max_vspeed_alt = altitudes['max_vspeed'] * 0.3048

        # Find the actual GPS points
        start_candidates = df[np.abs(df['AGL'] - turn_start_alt) <= (10 * 0.3048)]
        end_candidates = df[np.abs(df['AGL'] - max_vspeed_alt) <= (10 * 0.3048)]

        if len(start_candidates) > 0 and len(end_candidates) > 0:
            # Get the closest matches
            start_idx = start_candidates.iloc[np.abs(start_candidates['AGL'] - turn_start_alt).argmin()].name
            end_idx = end_candidates.iloc[np.abs(end_candidates['AGL'] - max_vspeed_alt).argmin()].name

            print(f"\n🔄 TURN SEGMENT ANALYSIS:")
            print(f"   Turn start: idx {start_idx} ({df.loc[start_idx, 'AGL']/0.3048:.1f}ft)")
            print(f"   Turn end: idx {end_idx} ({df.loc[end_idx, 'AGL']/0.3048:.1f}ft)")

            if start_idx < end_idx:
                turn_segment = df.loc[start_idx:end_idx]
                headings = turn_segment['heading'].values

                if len(headings) >= 2:
                    start_hdg = headings[0]
                    end_hdg = headings[-1]

                    # Try different rotation calculations
                    net_rotation = end_hdg - start_hdg
                    while net_rotation > 180:
                        net_rotation -= 360
                    while net_rotation < -180:
                        net_rotation += 360

                    # Calculate total rotation (cumulative)
                    total_rotation = 0
                    for i in range(1, len(headings)):
                        diff = headings[i] - headings[i-1]
                        while diff > 180:
                            diff -= 360
                        while diff < -180:
                            diff += 360
                        total_rotation += abs(diff)

                    # Get gswoop's published rotation
                    gswoop_rotation_match = re.search(r'degrees of rotation:\s+(\d+)\s+deg\s+\((\w+)-hand\)', result.stdout)
                    if gswoop_rotation_match:
                        gswoop_deg = float(gswoop_rotation_match.group(1))
                        gswoop_dir = gswoop_rotation_match.group(2)
                        gswoop_rotation = gswoop_deg if gswoop_dir == 'right' else -gswoop_deg
                    else:
                        gswoop_rotation = 0

                    print(f"   Start heading: {start_hdg:.1f}°")
                    print(f"   End heading: {end_hdg:.1f}°")
                    print(f"   Net rotation: {net_rotation:.1f}°")
                    print(f"   Total rotation: {total_rotation:.1f}°")
                    print(f"   gswoop rotation: {gswoop_rotation:.1f}°")

                    # Check which calculation matches gswoop
                    net_diff = abs(net_rotation - gswoop_rotation)
                    total_diff = abs(total_rotation - abs(gswoop_rotation))

                    print(f"   Net vs gswoop diff: {net_diff:.1f}°")
                    print(f"   Total vs gswoop diff: {total_diff:.1f}°")

                    if net_diff < total_diff and net_diff < 30:
                        print(f"   💡 gswoop likely uses NET rotation")
                    elif total_diff < net_diff and total_diff < 30:
                        print(f"   💡 gswoop likely uses TOTAL rotation")
                    else:
                        print(f"   ❓ gswoop calculation method unclear")

            else:
                print(f"   ❌ Invalid segment: start_idx ({start_idx}) >= end_idx ({end_idx})")

if __name__ == "__main__":
    # Debug the problematic case
    debug_gswoop_boundaries(os.path.expanduser("~/FlySight/Training/25-02-20/25-02-20-sw1.csv"))